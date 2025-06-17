# Anki deck for Mandarin Chinese listening practice

## Introduction

This Anki deck uses recordings of native speakers to practise Mandarin listening comprehension. Each card shows a transcription of a sentence (in simplified Chinese) with one word obscured and plays the corresponding recording, and you have to see if you can tell what word the person is saying. This type of card is known as a '[cloze](https://docs.ankiweb.net/editing.html#cloze-deletion)'.

The deck is designed for intermediate learners who are already comfortable reading and recognizing everyday words and want to practise listening to native speakers.

## Recordings and transcriptions

The data come from the [MAGICDATA Mandarin Chinese Read Speech Corpus](https://openslr.org/68), which in its entirety contains '755 hours of scripted read speech data from 1080 native speakers of the Mandarin Chinese spoken in mainland China'. The corpus is designed for training machine learning models, so it is very large. This deck only makes use of the `dev` and `test` sets, though it should work just as well on the full data set.

The data sets used comprise roughly 36,000 sentences. The script uses heuristics based on length and content to filter out unsuitable sentences, leaving approximately 15,000 cards in the deck.

## Word frequency

Word frequency data come from hermitdave's [FrequencyWords](https://github.com/hermitdave/FrequencyWords/).

## HSK

Each card includes the cloze word's [HSK](https://en.wikipedia.org/wiki/Hanyu_Shuiping_Kaoshi) (汉语水平考试) level, if it has one. The HSK word lists come from krmanik's [HSK repository](https://github.com/krmanik/HSK-3.0), which is based on the latest HSK levels from 2021 (levels 1–9, not just 1–6).

## Definitions

Definitions come from the ever-reliable [CC-CEDICT](https://cc-cedict.org/wiki/). Each card includes all the definitions for its cloze word.

## Pinyin

Sentences are split into words using [jieba](https://github.com/fxsjy/jieba) and converted from characters to pinyin with [pypinyin](https://github.com/mozillazg/python-pinyin). 

Both of these processes are somewhat problematic. 

Jieba does a commendable job, but sometimes segments sentences wrongly and results in non-words. I get around this by only choosing cloze words that are a) in the top 100,000 most frequent words and b) in the dictionary.

Pypinyin is not smart enough to use sentence context to improve its predictions, so in cases where a word can have multiple pronunciations, it always picks the most frequent. This means that it knows that the 重 in 重要 is not the same as the 重 in 重庆, but it can't tell whether 为 as a single word is wèi or wéi, and in fact always chooses the former.

If something seems fishy about the segmentation or the pinyin, I recommend using the 'Look up sentence' link on the card, which takes you to a breakdown of the sentence on the [MDBG Chinese dictionary](https://www.mdbg.net/chinese/dictionary).

## Translations

Reliable machine translation is tricky. I originally intended for this script to run entirely offline, so I sought machine translation tools that work without sending data to a server.

[Argos Translate](https://github.com/argosopentech/argos-translate) is the most popular Python library for 'conventional' offline translation. It follows the same principle as Google Translate and the like, but the results it produces are worse, because it's trained on less data and runs on a personal computer, rather than a server. The translations it produces are often acceptable, and it comes up with them remarkably quickly (more than five per second on my laptop). However, it tends to produce incomplete translations. When given a Chinese sentence that really means 'How's the weather in Wuhan?', it might translate it as just 'How's the weather?'

This problem with Argos Translate prompted (!) me to look for a better solution. While I have generally avoided using LLMs for anything, I have heard that they are good at translating. Being keen on running things offline and not wanting to pay to use an API for a hobby project, I installed [Ollama](https://ollama.com/) and tried out a few different models, landing on [llama-translate](https://ollama.com/7shi/llama-translate:8b-q4_K_M). This model's problems go in the other direction to Argos Translate's: while Argos Translate will miss words out, llama-translate will sometimes dream up extra words in a sentence it's translating, or even not translate the sentence at all and say something else. Asked to translate 'How's the weather in Wuhan?', it might respond 'The weather in Wuhan tomorrow is 28 degrees and sunny...', thus not doing what it was asked and fabricating information whole-cloth (it's not connected to the internet, so it has no way of knowing the true answer).

llama-translate has a higher success rate than Argos Translate, but the magnitude of its mistakes led me to devise a series of heuristics aimed at detecting inappropriate 'translations'. I was not satisfied with this situation, so I decided to renege on the idea of doing everything offline. As it stands, the translations in the deck were obtained by using [Selenium](https://www.selenium.dev/) to put the sentences through [DeepL](https://www.deepl.com).
