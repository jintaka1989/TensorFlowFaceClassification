Using these, You can create your own image data set for TensorFlow.

Run,
python main.py
after put classified jpg in assingd directory and
write "num_classes=****" to config.ini

///////////////For Example/////////////////
data_set
├── test
│   ├── class0
│   │   ├── search.jpg
│   │   ├── th_4.jpg
│   │   └── th_5.jpg
│   ├── class1
│   │   ├── search.jpg
│   │   ├── search_2.jpg
│   │   └── th_8.jpg
│   ├── class2
│   │   ├── search.jpg
│   │   ├── search_2.jpg
│   │   └── th_6.jpg
│   ├── class3
│   │   ├── search.jpg
│   │   ├── search_2.jpg
│   │   └── th_7.jpg
│   ├── class4
│   │   ├── th.jpg
│   │   ├── th_10.jpg
│   │   └── th_9.jpg
│   └── class5
│       ├── imgres.jpg
│       ├── imgres_2.jpg
│       └── imgres_8.jpg
└── train
    ├── class0
    │   ├── 1.jpg
    │   ├── 8.jpg
    │   └── 9.jpg
    ├── class1
    │   ├── 1.jpg
    │   ├── th_20.jpg
    │   └── th_9.jpg
    ├── class2
    │   ├── th.jpg
    │   ├── th_10.jpg
    │   └── th_9.jpg
    ├── class3
    │   ├── th.jpg
    │   ├── th_10.jpg
    │   └── th_9.jpg
    ├── class4
    │   ├── search.jpg
    │   ├── search_2.jpg
    │   └── th_9.jpg
    └── class5
        ├── test.jpg
        ├── red-sports-car.jpg
        └── redferrari.jpg
////////////////////////////////////////////

//////Detail/////////
■data_set
  −train.txt
    ・you have to write the location path of train data_set in this
    ・and write Classification Number after " "
    ・and write "\n"
    ・for example
-------train.txt-----------
data_set/test001.jpeg 0
data_set/test002.jpeg 1
data_set/test003.jpeg 2
data_set/test004.jpeg 3
---------------------------
  −test.txt
    ・you have to write the location path of train data_set in this in the same way as "train.txt"

■read_data.py
  To run this, you can get "models/model.ckpt". That ckpt file could be used to "use_model.py"

■use_model.py
  To run this, you can test "models/model.ckpt"(Using test data_set).

■create_data_set.py
after to run this,
cat class0.txt class1.txt class2.txt class3.txt class4.txt > train.txt
cat class0.txt class1.txt class2.txt class3.txt class4.txt > test.txt

you can't use " " in jpg filename
sudo rename "s/ /_/g" *.jpg
/////////////////////
