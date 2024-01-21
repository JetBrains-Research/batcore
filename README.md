# BaT CoRe: Baselines and testing framework for code reviewer recommendations

## Documentation

Extensive documentation for Bat CoRe can be found [here](https://batcore.readthedocs.io/en/latest/).
Bat CoRe can be installed via [pip](https://pypi.org/project/batcore/0.1.1/):

```pip install batcore```

## Framework
This repository provides a framework for testing code review recommendation algorithms.


### Model creation
* **RecommenderBase** is a simple interface for a generic recommender algorithm. It has a `fit` method that trains model on a given data and a `predict` method that predicts reviewers for the given pulls.
* **BanRecommenderBase** is an in interface derived from RecommenderBase that implements by candidate filtering by their activity and ownership of the pull request

### Dataset creation

* **DatasetBase** is an interface that encapsulates any preprocessing needed for the dataset with a method `preprocess`. 
* **GerritLoader** is a class that deals with the data loader from the PullExtractor tool: loads it, reformats it, and performs some high-level preprocessing (empty data removal, bot removal, alias matching, user encoding)
* **StandardDataset** an implementation of DatasetBase. Receives GerritLoader object and performs all the preprocessing that can be needed for a specific model
* **SpecialDatasets** has implementations of datasets that are unique to a specific recommender
* **get_gerrit_dataset** helping function for dataset creation. Dataset for a specific model can be created as `get_gerrit_dataset(gerritloader_object, model_cls=RecommenderBase_class)`

* **StreamLoaderBase** is an interface that encapsulates iteration over data. The interface views data as a temporal stream of events and 
yields pairs of consecutive segments of the stream - train (any set of events), test (a pull request event)

### Testing of models
* **TesterBase** is class that has `test_recommender` method. This method takes `RecommenderBase` and `StreamLoaderBase` and iterates over the dataset, retrains model on new train data, and calculates its predictions.
* **RecTester** an implementation of TesterBase that calculates metrics standard for recommendation systems (mrr, accuracy, etc)
* **SimulTester**  an implementation of TesterBase that tester for project-based metrics on a simulated history (*Mirsaeedi, Rigby, 2020*). Metrics for the simulated testing can be found in `Counter`.

An example of usage can be found in `example.py`
## Baselines

Framework contains implementations of the following models (`baselines\models`)

* RevFinder, *Who Should Review My Code?, Thongtanunam et al., 2015*
* Tie, *Who Should Review This Change?, Xin Xia et al., 2015*
* ACRec, *Who Should Comment on This Pull Request?, Jiang et al. 2017*
* cHRec, *Automatically Recommending Peer Reviewers in Modern Code Review, Zanjani et al., 2015*
* CN, *What Can We Learn from Code Review and Bug Assignment?, Yu et al., 2015*
* RevRec, *Search-Based Peer Reviewers Recommendation in Modern Code Review, Ouni et al., 2016*
* WRC, *Automatically Recommending Code Reviewers Based on Their Expertise: An Empirical Comparison, Hannebauer et al., 2016*
* xFinder, *Assigning change requests to software developers, Kagdi et al., 2011*
