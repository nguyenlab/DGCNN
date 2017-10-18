1. Generate data for Training - CV - Validation in Json format
+ AST ---> Graph: TreeData_IO.py
+ Manipulate with graph data (Virus database): GraphData_IO
+ CodeChef databases: CodeChef_Data.py
2. Construct networks
main_MultiChannelGCNN
3. Parameters: gcnn_params
- numDis = 300
- numOut = 104

- numView =1
- numCon =[30,600]
- datapath
4. Save Net Parameters (Weights, Token embedding)
- train_MultiChannelGCNN