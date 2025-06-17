import itertools
import math
from pathlib import Path
import os
import json
import bisect
import re
import random
import csv
import time

import jieba
from pypinyin import pinyin, Style
from pypinyin.contrib.tone_convert import to_tone
import argostranslate.package
import argostranslate.translate
from tqdm import tqdm
from ollama import chat
from ollama import ChatResponse
from selenium import webdriver
from selenium.webdriver.common.by import By
import selenium.common.exceptions


FROM_CODE = "zh"
TO_CODE = "en"
AUDIO_ORIG = Path.home() / "Downloads/chinesewav"
AUDIO_DEST = Path.home() / "chinesemp3s"
ANKI_MEDIA = Path.home() / ".local/share/Anki2/User 1/collection.media"
TRANSCRIPTIONS = ["TRANS_dev.txt", "TRANS_test.txt"]
JSON_FILE = "sentences.json"
FREQ_LIST_SIZE = 100000
CC_CEDICT = "cedict_ts.u8"
HSK_JSON = "hsk.json"
TRANSLATION_FILES = [
    "translated-deepl.json",
    "translated-llama.json",
    "translated-obt.json",
]


def cc_cedict_to_tone(string):
    """Convert a CC-CEDICT pinyin syllable to standard pinyin."""
    return to_tone(string.replace("u:", "ü"))


def cc_cedict_pinyin(string):
    """Convert CC-CEDICT headword pronunciation to standard pinyin.

    In CC-CEDICT syntax, the third field is the pronunciation in pinyin with
    tone numbers, where syllables are separated by spaces. This function
    converts from this format to standard pinyin with tone markings and words
    written together. CC-CEDICT contains proper nouns, where each word in the
    proper noun should be separated by a space. This function inserts a space
    before capital letters to account for this, yielding the following
    behaviour:

    dang1 ran2 -> dāngrán
    Mao2 Ze2 dong1 -> Máo Zédōng
    """
    tone_numbers = string.split()
    tone_markings = [cc_cedict_to_tone(word) for word in tone_numbers]
    return "".join(
        [tone_markings[0]]
        + [f" {word}" if word[0].isupper() else word for word in tone_markings[1:]]
    )


def numbers_to_markings(string):
    """Replace all occurrences of CC-CEDICT pinyin with standard pinyin."""
    reg = re.compile(r"[A-Za-z:]+\d")
    matches = reg.findall(string)
    new_string = string
    for match in matches:
        new_string = new_string.replace(match, cc_cedict_to_tone(match))
    return new_string


def pretty_meaning(string):
    """Mark up CC-CEDICT word meanings in semantic HTML."""
    reg = re.compile(r"\[.*?\]")
    matches = reg.findall(string)
    new_string = string
    for match in matches:
        contents = re.match(r"\[(.*)\]", match).group(1)
        replacement = f' <span class="pinyin">{numbers_to_markings(contents)}</span>'
        new_string = new_string.replace(match, replacement)
    reg = re.compile(r"([\u4e00-\u9fff]+)")
    return reg.sub(r'<span class="chinese">\1</span>', new_string)


def variant_char(simplified, definition):
    """Return True if `definition` describes a variant of `simplified`.

    CC-CEDICT is sorted by traditional character, but `read_dictionary` groups
    definitions by simplified character, and `cc_cedict_lookup` removes
    traditional characters. This means that some definitions lose context and
    become redundant. For example, one of the definitions of 个 is:

    箇 个 [ge4] /variant of 個|个[ge4]/

    This is useful information in the dictionary, but after we remove
    traditional characters, all it tells us it that 个 is a variant of itself.
    This function returns True for such definitions.
    """
    reg = re.compile(r"(old )?variant of (\w+\|)?(\w+)")
    if not reg.match(definition):
        return False
    return reg.match(definition).group(3) == simplified


def classifier(string):
    """Format a CC-CEDICT classifier indicator."""
    reg = re.compile("CL:(.*)")
    if not reg.match(string):
        return string
    split = reg.match(string).group(1).split(",")
    joined = ", ".join(split)
    return f"Classifier{'s' if len(split) > 1 else ''}: {joined}"


def remove_traditional(definition):
    """Remove traditional characters from a CC-CEDICT definition."""
    return re.sub(r"\w+\|(\w+)", r"\1", definition)


def read_dictionary():
    """Read `CC_CEDICT` and return a dictionary (!).

    In the dictionary, keys correspond to lists of dictionaries with pinyin and
    meanings.

    Read more about CC-CEDICT syntax: https://cc-cedict.org/wiki/syntax
    """
    reg = re.compile(r"(\S+) (\S+) \[(.*)\] /(.*)/$")
    dictionary = {}
    with open(CC_CEDICT, "r") as file:
        for line in file.readlines():
            result = reg.search(line)
            simplified = result.group(2)
            pronunciation = result.group(3)
            meanings = result.group(4).split("/")
            subdict = {"pinyin": pronunciation, "meanings": meanings}
            try:
                dictionary[simplified].append(subdict)
            except KeyError:
                dictionary.update({simplified: [subdict]})
    return dictionary


DICTIONARY = read_dictionary()


def cc_cedict_lookup(word):
    """Look up `word` in CC-CEDICT."""
    try:
        definitions = DICTIONARY[word]
    except KeyError:
        return None
    pronunciations = sorted(list(set([d["pinyin"] for d in definitions])))
    pron_dict = {}
    [pron_dict.update({p: []}) for p in pronunciations]
    for definition in definitions:
        for meaning in definition["meanings"]:
            if variant_char(word, meaning):
                continue
            pron_dict[definition["pinyin"]].append(
                pretty_meaning(remove_traditional(classifier(meaning)))
            )
    pronunciations = [
        {"pinyin": cc_cedict_pinyin(p), "meanings": pron_dict[p]}
        for p in pronunciations
        if pron_dict[p]
    ]
    return pronunciations


def format_definitions(definitions):
    """Format `definitions` into semantic HTML."""
    lines = []
    for definition in definitions:
        lines.append('<div class="pronunciation">')
        lines.append(f'<p><span class="pinyin">{definition["pinyin"]}</span></p>')
        lines.append("</div>")
        lines.append('<div class="meanings">')
        lines += [f"<p>{meaning}</p>" for meaning in definition["meanings"]]
        lines.append("</div>")
    return "".join(lines)


def read_hsk():
    """Read HSK word lists and return two dictionaries.

    The first dictionary associates a Chinese word of one or more characters
    with its HSK level, the second a single Chinese character with its HSK
    level.
    """
    hsk_dir = Path("hsk")
    chars_dir = hsk_dir / "hsk-chars"
    words_dir = hsk_dir / "hsk-words"
    levels = ["1", "2", "3", "4", "5", "6", "7-9"]
    chars = {}
    words = {}
    for (index, level) in reversed(list(enumerate(levels))):
        filename = f"HSK {level}.txt"
        with open(chars_dir / filename, "r") as file:
            [chars.update({line.strip(): index + 1}) for line in file.readlines()]
        with open(words_dir / filename, "r") as file:
            [words.update({line.strip(): index + 1}) for line in file.readlines()]
    return (words, chars)


(HSK_WORDS, HSK_CHARS) = read_hsk()


def hsk_level(string, char=False, if_none=None):
    dictionary = HSK_CHARS if char else HSK_WORDS
    try:
        return dictionary[string]
    except KeyError:
        return if_none


def true_hsk(string):
    return hsk_level(string, char=False) or hsk_level(string, char=True)


def sentence_difficulty(sentence, average=False):
    difficulties = [hsk_level(c, char=True, if_none=10) for c in sentence]
    if average:
        return sum(difficulties) / len(sentence)
    return max(difficulties)


argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(
        lambda x: x.from_code == FROM_CODE and x.to_code == TO_CODE, available_packages
    )
)
argostranslate.package.install_from_path(package_to_install.download())


def zh_to_en(sentence):
    """Use Argos Translate to translate from Chinese to English."""
    return argostranslate.translate.translate(sentence, FROM_CODE, TO_CODE)


def ollama_translate(sentence):
    """Use an LLM to translate from Chinese to English."""
    response: ChatResponse = chat(
        model="7shi/llama-translate:8b-q4_K_M",
        messages=[
            {"role": "user", "content": f"Translate Mandarin to English: {sentence}"}
        ],
    )
    return response["message"]["content"]


def translate_sentences(strings, outfile):
    """Translate `strings` from Chinese to English and write to `outfile`.

    Machine translation is a costly process, so this function checks that a
    sentence is not already present in `outfile` before translating it.
    """
    with open(outfile, "r") as file:
        responses = [json.loads(line)["zh"] for line in file.readlines()]
    responses.sort()
    for string in tqdm(strings):
        try:
            bisect_index(responses, string)
        except ValueError:
            bisect.insort(responses, string)
            translation = ollama_translate(string)
            with open(outfile, "a") as file:
                file.write(
                    json.dumps({"zh": string, "en": translation}, ensure_ascii=False)
                    + "\n"
                )


def acceptable_translation(pair):
    """Return True if `pair` seems an acceptable pair of translations.

    This function uses simple heuristics to estimate whether an English sentence
    is an acceptable translation of a Chinese sentence. It catches the most
    obvious offenders, but many unacceptable translations may slip through.
    """
    english = pair["en"]
    chinese = pair["zh"]
    if len(english.split()) > 1.5 * len(chinese):
        return False
    if len(english.split()) < 0.4 * len(chinese):
        return False
    if re.search("translat", english, re.IGNORECASE):
        return False
    # The Unicode range is the range of Chinese characters.
    patterns = ["Mandarin:", "English:", r"[\u4e00-\u9fff]", r"\[", r"\]"]
    for pattern in patterns:
        if re.search(pattern, english):
            return False
    return True


def improve_translations(outfile):
    """Filter out the worst translations in `outfile` and retranslate them."""
    with open(outfile, "r") as file:
        translations = [json.loads(line) for line in file.readlines()]
    acceptable = [trans for trans in translations if acceptable_translation(trans)]
    unacceptable = [
        trans for trans in translations if not acceptable_translation(trans)
    ]
    with open(outfile, "w") as file:
        file.writelines(
            [json.dumps(trans, ensure_ascii=False) + "\n" for trans in acceptable]
        )
    raw = [pair["zh"] for pair in unacceptable]
    translate_sentences(raw, outfile)


def read_translations():
    """Read `TRANSLATION_FILES` and return a dictionary of translations.

    The files in `TRANSLATION_FILES` are listed in order of priority. This
    function reads them in reverse order, so that translations from
    higher-priority lists overwrite those of lower priority.
    """
    paths = [Path("translations") / f for f in TRANSLATION_FILES]
    dictionary = {}
    for path in reversed(paths):
        with open(path, "r") as file:
            entries = [json.loads(line) for line in file.readlines()]
            [dictionary.update({entry["zh"]: entry["en"]}) for entry in entries]
    return dictionary


TRANSLATIONS = read_translations()


def lookup_translation(sentence):
    try:
        return TRANSLATIONS[sentence]
    except KeyError:
        return None


def characters_to_pinyin(sentence):
    """Read a Chinese sentence and return a dictionary.

    pypinyin is not smart enough to return different pinyin readings based on
    context, so using it to convert word-by-word (as here) gives the same
    results as converting an entire sentence in one go. A consequence of this is
    that a single character with multiple pinyin readings will always return the
    most frequent reading, e.g. 为 is always wèi and not wéi.
    """
    seg_list = list(jieba.cut(sentence))
    tone_markings = [
        "".join(list(itertools.chain.from_iterable(pinyin(item)))) for item in seg_list
    ]
    tone_numbers = [
        "".join(list(itertools.chain.from_iterable(pinyin(item, style=Style.TONE3))))
        for item in seg_list
    ]
    return {
        "segmented_characters": seg_list,
        "tone_markings": [tone_markings[0].capitalize()] + tone_markings[1:],
        "tone_numbers": [tone_numbers[0].capitalize()] + tone_numbers[1:],
    }


def mdbg_lookup(sentence):
    return (
        "https://www.mdbg.net/chinese/dictionary?page=worddict&wdrst=0&wdqtm=0&wdqcham=1&wdqt="
        f"{sentence}"
    )


def read_frequency():
    """Read a file of frequent words and return a dictionary.

    The list is extremely long and contains many (in my opinion) non-words, so
    this function only bothers with the first `FREQ_LIST_SIZE` words.
    """
    dictionary = {}
    with open("zh_cn_full.txt", "r") as file:
        split_lines = [line.split() for line in file.readlines()[:FREQ_LIST_SIZE]]
        [dictionary.update({line[0]: int(line[1])}) for line in split_lines]
    return dictionary


WORD_FREQ = read_frequency()


def word_frequency(word):
    try:
        return WORD_FREQ[word]
    except KeyError:
        return None


def rarest_word(words):
    """Return the rarest in a list of Chinese words.

    This function is interested only in words that appear in CC-CEDICT, so that
    it doesn't get distracted by non-words resulting from mis-segmentations.
    """
    in_dict = [word for word in words if cc_cedict_lookup(word)]
    lookups = [
        {"word": word, "frequency": word_frequency(word)}
        for word in in_dict
        if word_frequency(word)
    ]
    return min(lookups, key=lambda n: n["frequency"])


def length_poll(filename):
    """Read a file of Chinese sentences and graph their lengths.

    `filename` should be the name of a file of transcriptions of the sort
    provided by the MAGICDATA speech corpus.
    """
    with open(filename, "r") as file:
        sentences = [line.split()[2] for line in file.readlines()[1:]]
    lengths = [0] * 80
    for sent in sentences:
        try:
            lengths[len(sent)] += 1
        except IndexError:
            continue
    lengths = [round(math.log(length, 2)) if length != 0 else 0 for length in lengths]
    most = max(lengths)
    for row in range(most):
        for length in lengths:
            if length >= most - row:
                print("*", end="")
            else:
                print(" ", end="")
        print("")


class Sentence:
    """A recording and transcription of a Chinese sentence."""

    def __init__(self, info):
        """Read `info` and return a `Sentence`.

        `info` can be either a line from a transcription file or a JSON blob.
        """
        if isinstance(info, str):
            trans_line = info
            items = trans_line.split()
            self.filename = items[0]
            self.directory = items[1]
            self.audio_file = AUDIO_ORIG / self.directory / self.filename
            self.mp3_file = f"{Path(self.filename).stem}.mp3"
            self.sentence = items[2]
            processed_sentence = characters_to_pinyin(self.sentence)
            self.words = processed_sentence["segmented_characters"]
            self.tone_markings = processed_sentence["tone_markings"]
            self.tone_numbers = processed_sentence["tone_numbers"]
            self.translation = None
            try:
                rarest = rarest_word(self.words)
                self.rarest_word = rarest["word"]
                self.rarest_word_frequency = rarest["frequency"]
                self.rarest_word_meanings = cc_cedict_lookup(self.rarest_word)
            except ValueError:
                self.rarest_word = None
                self.rarest_word_frequency = None
                self.rarest_word_meanings = None
        if isinstance(info, dict):
            self.from_json(info)
        self.mp3_path = AUDIO_DEST / self.mp3_file
        if self.rarest_word:
            self.html_markup()
        self.mean_hsk = sentence_difficulty(self.sentence, average=True)
        self.max_hsk = sentence_difficulty(self.sentence)

    def __str__(self):
        return self.sentence

    def __repr__(self):
        return self.sentence

    def is_wanted(self, shortest, longest, unwanted_words, unwanted_beginnings):
        """Return True if `self` should be included in a card deck."""
        reg = re.compile(r"\[.*\]")
        if reg.search(self.sentence):
            return False
        if not self.rarest_word:
            return False
        if self.words[0] in unwanted_beginnings:
            return False
        intersection = set(self.words) & set(unwanted_words)
        if intersection:
            return False
        length = len(self.sentence)
        return not (length < shortest or length > longest)

    def cloze(self):
        cloze_word = self.rarest_word
        return self.sentence.replace(cloze_word, "{{c1::" + cloze_word + "}}")

    def play_wav(self):
        os.system(f"mpv {self.audio_file}")

    def play_mp3(self):
        os.system(f"mpv {self.mp3_path}")

    def wav_to_mp3(self):
        """Convert WAV to MP3, unless an MP3 with the same name exists.

        This generally reduces the filesize by about 10 times.
        """
        if self.mp3_path.exists():
            return
        bitrate = "16k"
        command = (
            "ffmpeg -hide_banner -loglevel error -i "
            f'"{self.audio_file}" -b:a {bitrate} "{self.mp3_path}"'
        )
        os.system(command)

    def translate(self):
        """Translate `self.sentence`, unless a translation exists already."""
        if self.translation:
            return
        lookup = lookup_translation(self.sentence)
        if lookup:
            self.translation = lookup
        else:
            self.translation = ollama_translate(self.sentence)

    def to_json(self):
        return {
            "file": self.mp3_file,
            "sentence": self.sentence,
            "cloze": self.cloze(),
            "marked_up": self.marked_up,
            "with_spaces": " ".join(self.words),
            "marked_up_words": " ".join(self.marked_words),
            "tone_markings": " ".join(self.tone_markings),
            "tone_numbers": " ".join(self.tone_numbers),
            "translation": self.translation,
            "rarest_word": self.rarest_word,
            "rarest_word_frequency": self.rarest_word_frequency,
            "rarest_word_meanings": self.rarest_word_meanings,
            "rarest_word_hsk": true_hsk(self.rarest_word),
            "mean_hsk": self.mean_hsk,
            "max_hsk": self.max_hsk,
        }

    def from_json(self, blob):
        self.mp3_file = blob["file"]
        self.sentence = blob["sentence"]
        self.words = blob["with_spaces"].split()
        self.tone_markings = blob["tone_markings"].split()
        self.tone_numbers = blob["tone_numbers"].split()
        self.translation = blob["translation"]
        self.rarest_word = blob["rarest_word"]
        self.rarest_word_frequency = blob["rarest_word_frequency"]
        self.rarest_word_meanings = blob["rarest_word_meanings"]

    def html_markup(self):
        """Mark the rarest word in the sentence.

        `self.rarest_word` is wrapped in <span class="sel"> in the Chinese
        sentence (characters and pinyin), so that it can be highlighted in the
        Anki card.
        """
        self.marked_up = self.sentence.replace(
            self.rarest_word, mark_sel(self.rarest_word)
        )
        word_indices = [
            index for (index, word) in enumerate(self.words) if word == self.rarest_word
        ]
        for index in word_indices:
            if "</span>" not in " ".join(self.tone_markings):
                tm_to_mark = self.tone_markings[index]
                self.tone_markings[index] = mark_sel(tm_to_mark)
                tn_to_mark = self.tone_numbers[index]
                self.tone_numbers[index] = mark_sel(tn_to_mark)
            self.marked_words = self.words.copy()
            word_to_mark = self.marked_words[index]
            self.marked_words[index] = mark_sel(word_to_mark)

    def hsk_label(self):
        level = true_hsk(self.rarest_word)
        if not level:
            return ""
        if level < 7:
            return str(level)
        if level == 7:
            return "7-9"

    def csv_row(self):
        return [
            f"[sound:{self.mp3_file}]",
            self.sentence,
            self.cloze(),
            self.marked_up,
            " ".join(self.words),
            " ".join(self.marked_words),
            " ".join(self.tone_markings),
            " ".join(self.tone_numbers),
            self.translation,
            self.rarest_word,
            self.rarest_word_frequency,
            format_definitions(self.rarest_word_meanings),
            self.hsk_label(),
            f"{self.mean_hsk:.3f}",
            self.max_hsk,
        ]


def mark_sel(string):
    return f'<span class="sel">{string}</span>'


def read_sentences(files, limit=None):
    all_lines = []
    all_sentences = []
    for filename in files:
        with open(filename, "r") as file:
            all_lines += file.readlines()[1:]
    for line in tqdm(all_lines[:limit]):
        all_sentences.append(Sentence(line))
    return all_sentences


def filter_sentences(sentences):
    """Remove unwanted sentences from `sentences`.

    More heuristics are involved here. Because of the way the speech corpus was
    collected, hundreds of the sentences are to do with song requests. The
    unwanted words are intended to remove most of these.
    """
    min_length = 6
    max_length = 12
    unwanted_words = ["播放", "听", "歌", "唱的歌", "歌曲", "首", "首歌曲"]
    unwanted_beginnings = ["放", "来", "请来"]
    filtered = [
        s
        for s in sentences
        if s.is_wanted(min_length, max_length, unwanted_words, unwanted_beginnings)
    ]
    return filtered


def process_sentences(sentences, outfile):
    """Write `sentences` to `outfile`.

    This function checks that each sentence is not already present in `outfile`,
    before converting its audio file and translating it. This may take a long
    time, depending on the method of translation used.

    `outfile` is a file with one JSON object per line.
    """
    if not Path(outfile).exists():
        os.system(f'touch "{Path(outfile).absolute()}"')
    with open(outfile, "r") as file:
        data = [json.loads(line) for line in file.readlines()]
    mp3s = sorted([item["file"] for item in data])
    print("Translating and converting audio...")
    for sentence in tqdm(sentences):
        try:
            bisect_index(mp3s, sentence.mp3_file)
        except ValueError:
            bisect.insort(mp3s, sentence.mp3_file)
            sentence.translate()
            sentence.wav_to_mp3()
            with open(outfile, "a") as file:
                file.write(json.dumps(sentence.to_json(), ensure_ascii=False) + "\n")


def bisect_index(sorted_list, item):
    """Return the index of `item` in `sorted_list` with a binary search."""
    index = bisect.bisect_left(sorted_list, item)
    if index != len(sorted_list) and sorted_list[index] == item:
        return index
    raise ValueError


def load_and_process(limit=None):
    """Load, filter, and process sentences."""
    print("Reading transcribed sentences...")
    sentences = read_sentences(TRANSCRIPTIONS, limit)
    filtered = filter_sentences(sentences)
    process_sentences(filtered, JSON_FILE)
    return filtered


def shuffle_sections(sorted_list, interval):
    """Shuffle `sorted_list` in sections of length `interval`.

    There are many rarest words that appear multiple times in the corpus. I want
    the deck to be sorted roughly by frequency of rarest words, but I don't want
    all the sentences with 的 to come in a clump at the start, so I use this
    function to shuffle them about a bit while maintaining the general trend.
    """
    output = []
    starts = list(range(0, len(sorted_list), interval))
    for start in starts:
        current_interval = min(interval, len(sorted_list) - start)
        output += random.sample(
            sorted_list[start : min(len(sorted_list), start + interval)],
            current_interval,
        )
    return output


def jsonlines_to_csv(jsonfile, csvfile):
    """Read sentences from `jsonfile` and write them to `csvfile`."""
    with open(jsonfile, "r") as file:
        sentences = [Sentence(json.loads(line)) for line in file.readlines()]
    sentences.sort(key=lambda s: s.rarest_word_frequency, reverse=True)
    sentences = shuffle_sections(sentences, 1000)
    with open(csvfile, "w", newline="") as file:
        writer = csv.writer(file, delimiter=",", quoting=csv.QUOTE_ALL)
        [writer.writerow(s.csv_row()) for s in sentences]


def deepl_translate(sentences, outfile):
    """Translate `sentences` on DeepL.com and write to `outfile`.

    As is generally the case with web scraping solutions using Selenium, this
    function will stop working if DeepL change the design of their website.
    """
    already_translated = []
    if Path(outfile).exists():
        with open(outfile, "r") as file:
            already_translated += [json.loads(line)["zh"] for line in file.readlines()]
    already_translated.sort()
    filtered = []
    for sentence in sentences:
        try:
            bisect_index(already_translated, sentence)
        except ValueError:
            filtered.append(sentence)
    if not filtered:
        return
    driver = webdriver.Firefox()
    driver.get("https://www.deepl.com/en/translator#zh/en-gb/")
    time.sleep(5)
    input_box = driver.find_element(By.XPATH, "//div[@_d-id='1']")
    output_box = driver.find_element(By.XPATH, "//div[@_d-id='6']")
    last_output = ""
    for sentence in tqdm(filtered):
        try:
            input_box.clear()
        except selenium.common.exceptions.InvalidElementStateException:
            deepl_translate(sentences, outfile)
            break
        input_box.send_keys(sentence)
        time.sleep(1.5)
        output = last_output
        while output == last_output:
            output = output_box.text
            time.sleep(0.1)
        last_output = output
        with open(outfile, "a") as file:
            file.write(
                json.dumps({"zh": sentence, "en": output}, ensure_ascii=False) + "\n"
            )
    driver.quit()
