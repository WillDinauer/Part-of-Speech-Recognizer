# Part of Speech Recognizer

This project trains a model to tag the part of speech of each word in a sentence, built using Java.

### Training

The following tags are tracked:
```
Tag     |   Part of Speech   |  Examples
------------------------------------------------------------
ADJ	    adjective	        new, good, high, special, big, local
ADV	    adverb	        really, already, still, early, now
CNJ	    conjunction	        and, or, but, if, while, although
DET	    determiner	        the, a, some, most, every, no
EX	    existential	        there, there's
FW	    foreign word        dolce, ersatz, esprit, quo, maitre
MOD	    modal verb	        will, can, would, may, must, should
N	    noun	        year, home, costs, time, education
NP	    proper noun	        Alison, Africa, April, Washington
NUM	    number	        twenty-four, fourth, 1991, 14:24
PRO	    pronoun	        he, their, her, its, my, I, us
P	    preposition	        on, of, at, with, by, into, under
TO	    the word to	        to
UH	    interjection        ah, bang, ha, whee, hmpf, oops
V           verb	        is, has, get, do, make, see, run
VD	    past tense	        said, took, told, made, asked
VG	    present participle  making, going, playing, working
VN	    past participl      given, taken, begun, sung
WH	    wh determiner       who, which, when, what, where, how
```

To train the model, we go through each sentence and count up, for each tag, how many times each word was observed with that tag, and how many times each other tag followed it. Thus, we construct two maps: one tracking transitions, and another tracking observations.
Based on the frequency of these transitions and observations, we can construct a Hidden Markov Model (HMM) to tag new, unseen sentences.

The following is an HMM constructed from tagged sentences. Tagged sentences are of the form 'Word/TAG [Word/TAG]...'
```
I/PRO fish/V
Will/NP eats/V the/DET fish/N
Will/MOD you/PRO cook/V the/DET fish/N
One/DET cook/N uses/VD a/DET saw/N
A/DET saw/N has/V many/DET uses/N
You/PRO saw/VD me/PRO color/V a/DET fish/N
Jobs/NP wore/VD one/DET color/N
The/DET jobs/N were/VD mine/PRO
The/DET mine/N has/V many/DET fish/N
You/PRO can/MOD cook/V many/PRO
```
![HMM](https://github.com/WillDinauer/Part-of-Speech-Recognizer/assets/77174175/82e909ee-f6e1-4125-b7fa-a15f23c829d1)

### Testing

To assess how good a model is, we compute how many tags it gets right and how many it gets wrong on some test sentences. Tests files are located in the `inputs/` folder.
Sentences are tagged using the trained model and applying the Viterbi Algorithm.

Each test tracks the number of correct and incorrect tags, reporting the accuracy for the given training and test set. For example, here are the results for the `Simple` and `Brown` text sets:

![TextTests](https://github.com/WillDinauer/Part-of-Speech-Recognizer/assets/77174175/074cc05c-6fb5-4e51-a7f1-d0c74c6562aa)
