01. Language Model
------------------
	In this part, we create our own bigram, unigram, trigram model and we teach them with 60 percent of our data. We use the chain rule probability theorem in this model.
		
	we train our models with get_train_set(...) function





0.2 Add-One (Laplace) Smoothing
-------------------------------
	In this part, We develop our model by a method that called add-one smoothing method and when we meet a word that is not our train, we use a method that add 1 our probabikity for our probability will not be resulted 0 






0.3 Generating E-Mails
----------------------
	In this section, we produce both unsmoothing and smoothin form of our language model and calculate the probabilities of these emulations.The words here are not more than 30, so we multiply their probability by multiplying rather than collecting their log.


	for unsmoothing model;
		get_nonsmooting_unigram_sentence(...)
		get_nonsmoothing_bigram_sentence(...)
		get_nonsmoothing_trigram_sentence(...)

	for smoothing model;
		get_smoothing_unigram_sentence(...)
		get_smoothing_bigram_sentence(...)
		get_smoothing_trigram_sentence(...)

We generate with using these functions.

Example Output:
---------------
NonSmoothing Unigram Sentences
	5.640446580917611e-18		specials grigsby/hou/ect@ect rights i !
Smoothing Unigram Sentences
	1.880932555399709e-29		issue doing meal allen , _ : plants : .
NonSmoothing Bigram Sentences
	4.718468487779554e-13		the combination was an important .
Smoothing Bigram Sentences
	1.1458117852480452e-81		costs of california problem with you started but we traded monday to date to changes for your phone .
NonSmoothing Trigram Sentences
	3.67877763169042e-12		what is the way , it says about easements .
Smoothing Trigram Sentences
	6.278900250838521e-61		``given what has beenhappening in and out very different issues from different traders .




0.4 Evaluation
--------------
	In this section, we create a perplexity account for each email in our test set with our "smoothing bigram model" and "smoothing trigram model" and we set up the probabilities of these emails but since the probabilities of these mails are underflow, we have to show them by taking their logs.



Example output:
---------------

Sentence:		can you please grant access to robert badeer for tagg/erms .   we need to get him access to tds and this requires a tagg login .   bob will be trading on the nw desk under mike grigsby effective monday march 11th .   can you please process this request by tomorrow .   please let me know if you have any questions .   thanks . pl

LoG Probability:			-1099.7213682671904
Perplexity Smoothing Bigram:		16591
Perplexity Smoothing Trigram:		20102

