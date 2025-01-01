using MyMachineLearningLibrary;

/*
 TODO:
	Rename variables, write better comments
	Currently, Save and Load works by making all variables public methods, I should look into changing this
	Add Softmax Activation and Categorical Cross Entropy Loss Function
 */



/*
 Create a ficticious dataset by randomly generating 10 numbers then classifying those numbers with a 0 if the sum would be less than .5 or 1 otherwise
 Split the dataset into a tarining and test set





 */

//const int SIZE_OF_DATA_SET = 1000;
//const int SIZE_OF_TRAINING_DATA = (int)(SIZE_OF_DATA_SET * .8);
//const int SIZE_OF_TESTING_DATA = SIZE_OF_DATA_SET - SIZE_OF_TRAINING_DATA;

//const int NUMBER_OF_INPUTS = 10;

//var trainingX = new double[SIZE_OF_TRAINING_DATA][];
//var trainingY = new double[SIZE_OF_TRAINING_DATA][];
//var testingX = new double[SIZE_OF_TESTING_DATA][];
//var testingY = new double[SIZE_OF_TESTING_DATA][];
//var rng = new Random();

//for(int i = 0; i < SIZE_OF_DATA_SET; i++)
//{
//	if(i < SIZE_OF_DATA_SET * .8)
//	{
//		trainingX[i] = new double[NUMBER_OF_INPUTS];

//		for (int j = 0; j < NUMBER_OF_INPUTS; j++)
//		{
//			trainingX[i][j] = rng.NextDouble();
//		}

//		trainingY[i] = new double[2];
//		trainingY[i][0] = trainingX[i].Sum() < 5 ? 0 : 1;
//		trainingY[i][1] = trainingX[i].Sum() < 5 ? 0 : 1;
//	}
//	else
//	{
//		testingX[i - SIZE_OF_TRAINING_DATA] = new double[NUMBER_OF_INPUTS];

//		for (int j = 0; j < NUMBER_OF_INPUTS; j++)
//		{
//			testingX[i - SIZE_OF_TRAINING_DATA][j] = rng.NextDouble();
//		}

//		testingY[i - SIZE_OF_TRAINING_DATA] = new double[2];
//		testingY[i - SIZE_OF_TRAINING_DATA][0] = testingX[i - SIZE_OF_TRAINING_DATA].Sum() < 5 ? 0 : 1;
//		testingY[i - SIZE_OF_TRAINING_DATA][1] = testingX[i - SIZE_OF_TRAINING_DATA].Sum() < 5 ? 0 : 1;
//	}

//}

//var nn = new NeuralNetwork(NUMBER_OF_INPUTS, .0001, new BinaryCrossEntropyLossFunction());
//nn.AddLayer(new DenseLayer(20, new SigmoidActivationFunction()));
//nn.AddLayer(new DenseLayer(15, new SigmoidActivationFunction()));
//nn.AddLayer(new DenseLayer(10, new SigmoidActivationFunction()));
//nn.AddLayer(new DenseLayer(5, new SigmoidActivationFunction()));
//nn.AddLayer(new DenseLayer(2, new SoftmaxActivationFunction()));

//nn.Train(1000, trainingX, trainingY, 1, true);

//nn.Save("test.json");

//var nn2 = NeuralNetwork.Load("test.json");
//var accuracy = nn2.Test(testingX, testingY);

//Console.WriteLine($"Test Accuracy: {accuracy}");

/*
 
 Creating Flush Detection model and saving it
 
 
 
 */



//int[] data = new int[52];
//for (int i = 0; i < data.Length; i++)
//{
//	data[i] = i % 4;
//}

//const int TRAINING_COUNT = 10000;
//const int TESTING_COUNT = (int)(TRAINING_COUNT * .2);

//var trainingX = new double[TRAINING_COUNT][];
//var testingX = new double[TESTING_COUNT][];

//var trainingY = new double[TRAINING_COUNT][];
//var testingY = new double[TESTING_COUNT][];

//var rng = new Random();

//for(int i = 0; i < TRAINING_COUNT; i++)
//{
//	double[] x = new double[5];

//	if (rng.NextDouble() < .5)
//	{
//		int r = rng.Next(4);

//		for (int j = 0; j < x.Length; j++)
//		{
//			x[j] = r / 3.0;
//		}
//	}
//	else
//	{
//		int[] selections = new int[x.Length];
//		for (int j = 0; j < selections.Length; j++)
//		{
//			int t = rng.Next(data.Length);
//			while (selections.Any(s => s == t))
//			{
//				t = rng.Next(data.Length);
//			}

//			selections[j] = t;

//			x[j] = data[t] / 3.0;
//		}
//	}

//	trainingX[i] = x;

//	bool match = true;
//	for (int j = 0; j < x.Length - 1; j++)
//	{
//		if (x[j] != x[j + 1])
//		{
//			match = false;
//			break;
//		}
//	}

//	trainingY[i] = new double[1];
//	trainingY[i][0] = match ? 1 : 0;
//}

//for (int i = 0; i < TESTING_COUNT; i++)
//{
//	double[] x = new double[5];

//	if (rng.NextDouble() < .5)
//	{
//		int r = rng.Next(4);

//		for (int j = 0; j < x.Length; j++)
//		{
//			x[j] = r / 3.0;
//		}
//	}
//	else
//	{
//		int[] selections = new int[x.Length];
//		for (int j = 0; j < selections.Length; j++)
//		{
//			int t = rng.Next(data.Length);
//			while (selections.Any(s => s == t))
//			{
//				t = rng.Next(data.Length);
//			}

//			selections[j] = t;

//			x[j] = data[t] / 3.0;
//		}
//	}

//	testingX[i] = x;

//	bool match = true;
//	for (int j = 0; j < x.Length - 1; j++)
//	{
//		if (x[j] != x[j + 1])
//		{
//			match = false;
//			break;
//		}
//	}

//	testingY[i] = new double[1];
//	testingY[i][0] = match ? 1 : 0;
//}

//var nn = new NeuralNetwork(5, .01, new BinaryCrossEntropyLossFunction());
//nn.AddLayer(new DenseLayer(6, new SigmoidActivationFunction()));
//nn.AddLayer(new DenseLayer(1, new SigmoidActivationFunction()));

//nn.Train(5000, trainingX, trainingY, 1, true);

//Console.WriteLine();
//Console.WriteLine(nn.Test(testingX, testingY));

//nn.Save("flush_detection_model.json");


/*
 Loading and testing flush detection model
 
 
 
 
 
 */


//int[] data = new int[52];
//for (int i = 0; i < data.Length; i++)
//{
//	data[i] = i % 4;
//}

//const int TRAINING_COUNT = 10000;
//const int TESTING_COUNT = (int)(TRAINING_COUNT * .2);

//var trainingX = new double[TRAINING_COUNT][];
//var testingX = new double[TESTING_COUNT][];

//var trainingY = new double[TRAINING_COUNT][];
//var testingY = new double[TESTING_COUNT][];

//var rng = new Random();

//for (int i = 0; i < TESTING_COUNT; i++)
//{
//	double[] x = new double[5];

//	if (rng.NextDouble() < .5)
//	{
//		int r = rng.Next(4);

//		for (int j = 0; j < x.Length; j++)
//		{
//			x[j] = r / 3.0;
//		}
//	}
//	else
//	{
//		int[] selections = new int[x.Length];
//		for (int j = 0; j < selections.Length; j++)
//		{
//			int t = rng.Next(data.Length);
//			while (selections.Any(s => s == t))
//			{
//				t = rng.Next(data.Length);
//			}

//			selections[j] = t;

//			x[j] = data[t] / 3.0;
//		}
//	}

//	testingX[i] = x;

//	bool match = true;
//	for (int j = 0; j < x.Length - 1; j++)
//	{
//		if (x[j] != x[j + 1])
//		{
//			match = false;
//			break;
//		}
//	}

//	testingY[i] = new double[1];
//	testingY[i][0] = match ? 1 : 0;
//}

//var nn2 = NeuralNetwork.Load("flush_detection_model.json");

//Console.WriteLine(nn2.Test(testingX, testingY));

/*
 
 Straight Detection model training and saving
 
 
 
 */


//int[] data = new int[52];
//for (int i = 0; i < data.Length; i++)
//{
//	data[i] = i % 13;
//}

//const int TRAINING_COUNT = 10000;
//const int TESTING_COUNT = (int)(TRAINING_COUNT * .2);

//var trainingX = new double[TRAINING_COUNT][];
//var testingX = new double[TESTING_COUNT][];

//var trainingY = new double[TRAINING_COUNT][];
//var testingY = new double[TESTING_COUNT][];

//var rng = new Random();

//for (int i = 0; i < TRAINING_COUNT; i++)
//{
//	double[] x = new double[5];

//	if (rng.NextDouble() < .5)
//	{
//		int r = rng.Next(9);
//		double[] selections = new double[x.Length];

//		for (int j = 0; j < selections.Length; j++)
//		{
//			selections[j] = r + j;
//		}

//		for(int k = 0; k < 10; k++)
//		{
//			for (int j = selections.Length; j > 0; j--)
//			{
//				r = rng.Next(j);
//				var temp = selections[r];
//				selections[r] = selections[j - 1];
//				selections[j - 1] = temp;
//			}
//		}

//		for (int j = 0; j < selections.Length; j++)
//		{
//			selections[j] /= 12;
//		}

//		x = selections;
//	}
//	else
//	{
//		int[] selections = new int[x.Length];
//		for (int j = 0; j < selections.Length; j++)
//		{
//			int t = rng.Next(data.Length);
//			while (selections.Any(s => s == t))
//			{
//				t = rng.Next(data.Length);
//			}

//			selections[j] = t;

//			x[j] = data[t] / 12;
//		}
//	}

//	trainingX[i] = x;

//	var xCopy = x.ToList();
//	xCopy.Sort();

//	bool match = true;
//	for (int j = 0; j < xCopy.Count - 1; j++)
//	{
//		if (xCopy[j] + (1.0 / 12.0) != xCopy[j + 1])
//		{
//			match = false;
//			break;
//		}
//	}

//	trainingY[i] = new double[1];
//	trainingY[i][0] = match ? 1 : 0;
//}

//for (int i = 0; i < TESTING_COUNT; i++)
//{
//	double[] x = new double[5];

//	if (rng.NextDouble() < .5)
//	{
//		int r = rng.Next(9);
//		double[] selections = new double[x.Length];

//		for (int j = 0; j < selections.Length; j++)
//		{
//			selections[j] = r + j;
//		}

//		for (int k = 0; k < 10; k++)
//		{
//			for (int j = selections.Length; j > 0; j--)
//			{
//				r = rng.Next(j);
//				var temp = selections[r];
//				selections[r] = selections[j - 1];
//				selections[j - 1] = temp;
//			}
//		}

//		for (int j = 0; j < selections.Length; j++)
//		{
//			selections[j] /= 12;
//		}

//		x = selections;
//	}
//	else
//	{
//		int[] selections = new int[x.Length];
//		for (int j = 0; j < selections.Length; j++)
//		{
//			int t = rng.Next(data.Length);
//			while (selections.Any(s => s == t))
//			{
//				t = rng.Next(data.Length);
//			}

//			selections[j] = t;

//			x[j] = data[t] / 12;
//		}
//	}

//	testingX[i] = x;

//	var xCopy = x.ToList();
//	xCopy.Sort();

//	bool match = true;
//	for (int j = 0; j < xCopy.Count - 1; j++)
//	{
//		if (xCopy[j] + (1.0 / 12.0) != xCopy[j + 1])
//		{
//			match = false;
//			break;
//		}
//	}

//	testingY[i] = new double[1];
//	testingY[i][0] = match ? 1 : 0;
//}

//var nn = new NeuralNetwork(5, .01, new BinaryCrossEntropyLossFunction());
//nn.AddLayer(new DenseLayer(24, new SigmoidActivationFunction()));
//nn.AddLayer(new DenseLayer(1, new SigmoidActivationFunction()));

//nn.Train(5000, trainingX, trainingY, 1, true);

//Console.WriteLine();
//Console.WriteLine(nn.Test(testingX, testingY));

//nn.Save("straight_detection_model.json");








int[] data = new int[52];
for (int i = 0; i < data.Length; i++)
{
	data[i] = i % 13;
}

const int TESTING_COUNT = 50000;

var testingX = new double[TESTING_COUNT][];

var testingY = new double[TESTING_COUNT][];

var rng = new Random();

for (int i = 0; i < TESTING_COUNT; i++)
{
	double[] x = new double[5];

	if (rng.NextDouble() < .5)
	{
		int r = rng.Next(9);
		double[] selections = new double[x.Length];

		for (int j = 0; j < selections.Length; j++)
		{
			selections[j] = r + j;
		}

		for (int k = 0; k < 10; k++)
		{
			for (int j = selections.Length; j > 0; j--)
			{
				r = rng.Next(j);
				var temp = selections[r];
				selections[r] = selections[j - 1];
				selections[j - 1] = temp;
			}
		}

		for (int j = 0; j < selections.Length; j++)
		{
			selections[j] /= 12;
		}

		x = selections;
	}
	else
	{
		int[] selections = new int[x.Length];
		for (int j = 0; j < selections.Length; j++)
		{
			int t = rng.Next(data.Length);
			while (selections.Any(s => s == t))
			{
				t = rng.Next(data.Length);
			}

			selections[j] = t;

			x[j] = data[t] / 12;
		}
	}

	testingX[i] = x;

	var xCopy = x.ToList();
	xCopy.Sort();

	bool match = true;
	for (int j = 0; j < xCopy.Count - 1; j++)
	{
		if (xCopy[j] + (1.0 / 12.0) != xCopy[j + 1])
		{
			match = false;
			break;
		}
	}

	testingY[i] = new double[1];
	testingY[i][0] = match ? 1 : 0;
}

var nn2 = NeuralNetwork.Load("straight_detection_model.json");
Console.WriteLine(nn2.Test(testingX, testingY));