.. _datadesc_toplevel:

================
Data Description
================

Processing MRLoader output
==========================

To process the data loaded with `MRLoader <https://github.com/JetBrains-Research/MR-loader>`_ we provide ``MRLoaderData`` class.
``MRLoaderData`` processes data from the raw output and filters it from `from_date` to `to_date`:


.. code-block:: python

    from batcore.data import MRLoaderData

    data = MRLoaderData('path/to/the/directory/containing/output/of/MRLoader',
                         from_date=datetime(), # all events before are removed
                         to_date=datetime(), # all events after are removed
                        )


Resulting instance of ``MRLoaderData`` has three relevant fields: ``pulls``, ``comments``, and ``commits``.

Dataframes
----------
``pulls`` is a pandas DataFrame with information about pull requests. It has following fields:

* **key_change** - identifier of the pull request
* **file** - list of files' paths changed in the pull request
* **reviewer** - list of reviewers
* **date** - time of the creation of the pull request (datetime.datetime instance)
* **owner** - list of owners of the pull requests. We identify users by strings with the following format "{name}:{e-mail}:{login}".
* **title** - title of the pull request
* **status** - status of the pull request (one of three: "MERGED", "OPEN", and "ABANDONED")
* **closed** - date when pull request was closed
* **author** - set with authors of the pull request (i.e. authors of the commits)

``commits`` is a pandas DataFrame with information about commits. It has following fields:

* **key_commit** - identifier of the commit
* **key_change** - id of the pull request with the commit
* **key_file** - file path of the changed file (in case of several changed files, there will be several entries with same `key_commit`)
* **key_user** - author of the commit
* **date** - date of the commit

``comments`` is a pandas DataFrame with information about commits. It has following fields:

* **key_change** - id of the pull request with the commit
* **key_file** - file path of the changed file (nan when the comment was made not to the file)
* **key_user** - author of the commit
* **date** - date of the commit

Saving and Loading
------------------

You can save data from ``MRLoaderData`` with:

.. code-block:: python

    data.to_checkpoint('path/to/checkpoint')


And load with:

.. code-block:: python

    data = MRLoaderData().from_checkpoint('path/to/checkpoint')

Custom data
===========

``from_checkpoint`` method can also be used to load custom data.
Provided data should be in a specific format.
Path to checkpoint should lead to the folder with `pulls.csv` and optionally with `comments.csv` and `commits.csv`.
Those files should have a header with feature names from above and first index column.
Not all of the features are necessary for all of the models and some of them can be omitted and replaced with nan values.
All of the relevant columns must contain data in the form described above with the following exceptions:

* `dates` - date-string with the following format "yyyy-mm-dd hh:mm:ss"
* `user_ids` - can be anything, but if you use our implementation of alias matching it should be a string with the format "{name}:{e-mail}:{login}"


Required data for different models
----------------------------------

* **ACRec**
    * pulls: date, reviewer, key_change, owner
    * comments: key_change, key_user
* **cHRev**
    * pulls: date, reviewer, key_change, file, owner
    * comments: date, key_user, key_file
* **RevFinder**
    * pulls: date, reviewer, key_change, file, owner
* **RevRec**
    * pulls: date, reviewer, key_change, owner, file
    * comments: date, key_user, key_file, key_change
* **Tie**
    * pulls: date, reviewer, key_change, title, file, owner
* **WRC**
    * pulls: date, reviewer, key_change, file, owner
* **xFinder**
    * pulls: date, reviewer, key_change, owner, file
    * commits: date, key_user, key_file
* **CN**
    * pulls: date, reviewer, key_change, owner
    * comments: date, key_user, key_change


