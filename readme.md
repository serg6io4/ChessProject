## Installation

To run this project, you'll need to set up the required dependencies. Follow these steps:

1. Clone the repository:https://github.com/serg6io4/ChessProject

2. Create a virtual environment (optional but recommended):

```env
virtualenv venv
source venv/bin/activate
```

3. Install project dependencies:

```setup
pip install -r requirements.txt
```

4. Run the project(after introduce your image in the dataset):
```setup
python image2Fen.py "yourimagename.png"
```
python image2Fen.py "yourimagename.png"

## Training

To train the model(with your own data), you have to use the next command line:

```train
python train.py "path_to_data_directory" "path_to_save_directory"
```
>📋You have to run this command line. There are two arguments, the first is the directory of your main folder of data, and the second argument is the directory to save the model.

## Evaluation

To train and evaluate the model(with your own data), you have to use the next command line:

```eval
python eval.py "path_to_data_directory" "path_to_save_directory"
```
>📋This command line execute the training and evaluation of the classifier that will create with your own data, and show graphs like confusion matrix, accuracy and training and validation loss for epoch.

## Results 

My model achieves the following performance:

| Model name                         | Accuracy        | Training and validation loss |
|------------------------------------|-----------------|------------------------------|
| mobilenetv2_chess_classification   |     98%         |   0,0548      0,0302         |