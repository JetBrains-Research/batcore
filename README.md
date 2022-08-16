# rrr/rrr



## Getting Started

Download links:

SSH clone URL: ssh://git@git.jetbrains.team/rrr/framework.git

HTTPS clone URL: https://git.jetbrains.team/rrr/framework.git



These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

What things you need to install the software and how to install them.

```
Examples
```


## Framework
This repository provides a framework for testing code review recommendation algorithms. Our framework can be split into 4 main parts: TesterBase, RecommenderBase, DatasetBase and data IteratorBase.

* **TesterBase** is class that has `test_recommender` method. This method takes `RecommenderBase` and `IteratorBase` and iterates over the dataset to performs any calculates outputs of the recommender and desirable metrics.
  * We provide two different ready-to-use TesterBase implementation: a tester that calculates standard recommender metrics (e.g. accuracy, mrr) and tester for project-based metrics on a simulated history (*Mirsaeedi, Rigby, 2020*).
  * Metrics for the simulated testing can be found in `Counter`.
* **RecommenderBase** is a simple interface for a generic recommender algorithm. It has a `fit` method that trains model on a given data and a `predict` method that predicts reviewers for the given pulls.
* **DatasetBase** is an interface that encapsulates any preprocessing needed for the dataset with a method `preprocess`. 
  * Additionally, we provide classes for gathering data of github and gerrit projects that are mined with our tool.
  * For testing on simulated data dataset class also should implement method `replace`, which take datapoint and best predicted reviewer and replaces random real reviewer with a predicted one.
* **IteratorBase** is an interface that encapsulates iteration over data. Since in literature there are several ways to iterate we moved this logic to the separate class.
  * We provide two iterators from the box: one-by-one iterator over reviews and batch iterator that groups reviews by time.
## Baselines

We also we implemented several baselines algorithms within our framework. All implementations can be found in `baselines`.
Implemented baselines:

* RevFinder, *Who Should Review My Code?, Thongtanunam et al., 2015*
* Tie, *Who Should Review This Change? Xin Xia et al., 2015*
* *Expanding the Number of Reviewers, Chueshev et al. 2020*