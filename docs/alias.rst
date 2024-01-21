.. _alias_toplevel:

==============
Alias Matching
==============

This is an adaptation of the alias merging algorithm from `Mining Email Social Networks <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bird2006mes.pdf>`_.
The steps of this method is straightforward: the complete weighted graph is built with contributors as nodes.
The weight of each edge represents the similarity between two users based on the string comparison techniques.
After that, the graph is split into clusters based on the distance between nodes.

In the Gerrit data, contributors can have 3 features: name, login, and e-mail.


* **Name normalization.** We remove all punctuation, suffixes, and prefixes from the names. Additionally, titles and clarification terms ('admin', 'support') are removed. After that, the name is transformed into lowercase. Lastly, first and last names are calculated as the first and last part term after a split by space symbol.

* **Name similarity.** We measure the similarity between two names with a Levenstein distance normalized to $[0, 1]$ interval. The distance is calculated between full names, first names, and second names separately. The resulting name similarity is the minimum distance between full names and averaged distances between first and last names.

* **Name-email similarity.** The distance between the name and the email is a 0-1 measure. The distance is deemed to be 0 if and only if the e-mail consists of the full name or first and last name.

* **E-mail similarity.** For email similarity, we calculate normalized Levenstein distance between two bases (part of the e-mail preceding to @).

* **Login-email similarity.** Levenstein distance between the base of the e-mail and the login.

* **User identifier distance.** The resulting distance between two users is the minimum of the metrics above.

After the distance is calculated, we split users into clusters via `Agglomerative Clustering <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>`_ with a complete linkage.
All developers in one cluster are treated as the same and mapped to the one resulting id.

.. autofunction:: batcore.alias.get_clusters

.. autofunction:: batcore.alias.get_sim_matrix

.. autofunction:: batcore.alias.sim_users

.. autofunction:: batcore.alias.get_norm_levdist

.. autofunction:: batcore.alias.name_handle_dist