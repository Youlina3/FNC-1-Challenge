# FNC-1 Challenge
## This is my indivudual NLP course project (MSCI641,the University of Waterloo). Course Page: https://uwaterloo.ca/graduate-studies-academic-calendar/node/7481

###Project Introduction:<br /> 
The Project description has been adapted from the description on the FNC-1 website (http://www.fakenewschallenge.org). 

Fake news, defined by the New York Times as “a made-up story with an intention to deceive”1, often for a secondary gain, is arguably one of the most serious challenges facing the news industry today. In a December Pew Research poll, 64% of US adults said that “made-up news” has caused a “great deal of confusion” about the facts of current events2. 
 
The goal of the Fake News Challenge is to explore how artificial intelligence technologies, particularly machine learning and natural language processing, might be leveraged to combat the fake news problem. We believe that these AI technologies hold promise for significantly automating parts of the procedure human fact checkers use today to determine if a story is real or a hoax.


###My Approach:<br /> 
I implemented **four neural network models** to deal with this problem:  **LSTMs (Baseline), Bidirectional LSTMs (Baseline), LSTMs with attention and conditional encoding LSTMs with attention (CEA LSTMs)** (Pfohl et al., 2017). The final results show that CELA outperformed than the other three models, with the accuracy of 96.60% and the competition weighted score of 1819.50.  The second good model is LSTM with attention, with the accuracy 95.37% and the competition weighted score of 1817.0. Bidirectional LSTMs achieved 95.35% and the weighted score of 1156. Baseline LSTMs achieved the accuracy 90.61% and the weighted score of 1204.25. Furthermore, through the experiments, I found CEA LSTMs and LSTMs with attention performed well with long truncation length while the performance of baseline LSTM decreased as the truncation length increased. Moreover, I found the GloVe with basic LSTMs model outperformed than Word2Vec with basic LSTMs model.  

###Development Environment:<br />
The code for these four models was developed under the environment of Python 3.6.6, Keras 2.2.2, and Tensorflow 1.9.0.

###Reference:<br /> 
Fake News Challenge Stage 1 (FNC-1): Stance Detection http://www.fakenewschallenge.org <br /> 
FNC-1 Github repositories: https://github.com/FakeNewsChallenge <br />
FNC-1 official baseline github: https://github.com/FakeNewsChallenge/fnc-1-baseline <br />
Codalab: https://competitions.codalab.org/competitions/19111#phases <br />
Stephen Pfohl, Oskar Triebe, and Ferdinand Legros. 2017. Stance Detection for the Fake News Challenge with Attention and Conditional Encoding
 
