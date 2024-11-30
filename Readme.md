Two class classification metrics can be converted to 
multiclass classification

Let's take precision for example
There are 3 ways of doing this
- Macro averaged precision : calculate precision for all
classes individually and then average them
- Micro averaged precision : calculate class wise true positive
and false positive and then use that to calculate overall precision
- weighted precision : same as micro but in this case, it is weighted average 
depending on the number of items in each class

## Multilabel classification

-- We can use log loss for multilabel classification
you can convert the targets to binary format and then use a log loss for each column
In the end you can take average of log loss in each column. 
This is also known as mean column wise log loss
