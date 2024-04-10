导入节点时，参考如下代码

```cypher
LOAD CSV FROM "file:///countries.csv" AS line
create (a:country{CountryID:line[0],name:line[1],army:line[2],desc:line[3]})
```



导入关系时，参考如下代码

```cypher
LOAD CSV WITH HEADERS FROM 'file:///character2country.csv' AS row
MATCH (c:character {characterID: row.START_ID})
MATCH (co:country {CountryID: row.END_ID})
MERGE (c)-[r:from {type: row.type}]->(co);
```

其中第一行为从文件中读取第一行作为参数名，只有在使用了该参数后，才可以使用`line.name`这样的表示方式，否则需使用`line[0]`的表示方式。`row`为每行的别名

`MATCH (c:character {characterID: row.START_ID})`意味匹配所有节点标签为`character`（别名设置为`c`）的节点，其`characterID`属性对应`row`的`START_ID`，下一行同上

`MERGE`创建关系，从`c`到`co`，其中`r`为关系的别名，关系标签为`from`，关系的属性`type`的值为`row.type`
