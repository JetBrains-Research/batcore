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

    # or load from checkpoint
    # data = MRLoaderData().from_checkpoint('path/to/checkpoint')


Resulting instance of ``MRLoaderData`` has three relevant fields: ``pulls``, ``comments``, and ``commits``.

Dataframes
----------
**pulls** is a pandas DataFrame with information about pull requests. It has following fields:

* **key_change** - string identifier of the pull request
* **file** - list of files' paths changed in the pull request
* **reviewer** - list of reviewers
* **date** - time of the creation of the pull request (datetime.datetime instance)
* **owner** - list of owners of the pull requests. We identify users by strings with the following format "{name}:{e-mail}:{login}".
* **title** - title of the pull request
* **status** - status of the pull request (one of three: "MERGED", "OPEN", and "ABANDONED")
* **closed** - date when pull request was closed
* **author** - set with authors of the pull request (i.e. authors of the commits)

**commits** is a pandas DataFrame with information about commits. It has following fields:

* **key_commit** - identifier of the commit
* **key_change** - id of the pull request with the commit
* **key_file** - file path of the changed file (in case of several changed files, there will be several entries with same `key_commit`)
* **key_user** - author of the commit
* **date** - date of the commit

**comments** is a pandas DataFrame with information about commits. It has following fields:

* **key_change** - id of the pull request with the commit
* **key_file** - file path of the changed file (nan when the comment was made not to the file)
* **key_user** - author of the commit
* **date** - date of the commit