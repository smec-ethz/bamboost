---
title: Index troubleshooting
---

## Problems with sql database bamboost.db

If you have problems with the index you may delete it. 



## Regenerating the index 

If you just deleted the index or you tansfered databases that were not initially created on this filesystem, you may need to regenerate the index. 

```python
In [1]: from bamboost.index import DatabaseTable, IndexAPI, config
In [2]: idx = IndexAPI()
```

A typical symptom is an empty index table:
```python
In [7]: idx.read_table()
Out[7]: 
Empty DataFrame
Columns: [id, path]
Index: []

```

You can rediscover database this way
```python
In [8]: idx.scan_known_paths()
```

And now the table should not be empty anymore.

```python
In [9]: idx.read_table()
Out[9]: 
            id                                               path
0   730B3EEF6D  /cluster/work/cmbm/...
1   1A1F867229  /cluster/work/cmbm/...
```

If that didn't work make sure the appropriate paths are in the config file.
