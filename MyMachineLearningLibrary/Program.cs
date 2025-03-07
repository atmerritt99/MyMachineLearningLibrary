using MyMachineLearningLibrary;

/*
 TODO:
	Rename variables, write better comments
	Currently, Save and Load works by making all variables public methods, I should look into changing this
	Add Weight Initialization Methods
	Speed up matrix multiplication using parallel programming
 */



/*
 Create a ficticious dataset by randomly generating 10 numbers then classifying those numbers with a 0 if the sum would be less than .5 or 1 otherwise
 Split the dataset into a tarining and test set





 */

const int SIZE_OF_DATA_SET = 1000;
const int SIZE_OF_TRAINING_DATA = (int)(SIZE_OF_DATA_SET * .8);
const int SIZE_OF_TESTING_DATA = SIZE_OF_DATA_SET - SIZE_OF_TRAINING_DATA;

const int NUMBER_OF_INPUTS = 5;

var trainingX = new double[SIZE_OF_TRAINING_DATA][];
var trainingY = new double[SIZE_OF_TRAINING_DATA][];
var testingX = new double[SIZE_OF_TESTING_DATA][];
var testingY = new double[SIZE_OF_TESTING_DATA][];
var rng = new Random();

for (int i = 0; i < SIZE_OF_DATA_SET; i++)
{
	if (i < SIZE_OF_DATA_SET * .8)
	{
		trainingX[i] = new double[NUMBER_OF_INPUTS];

		for (int j = 0; j < NUMBER_OF_INPUTS; j++)
		{
			trainingX[i][j] = rng.NextDouble();
		}

		trainingY[i] = new double[1];
		trainingY[i][0] = trainingX[i].Sum() < NUMBER_OF_INPUTS / 2.0 ? 0 : 1;
	}
	else
	{
		testingX[i - SIZE_OF_TRAINING_DATA] = new double[NUMBER_OF_INPUTS];

		for (int j = 0; j < NUMBER_OF_INPUTS; j++)
		{
			testingX[i - SIZE_OF_TRAINING_DATA][j] = rng.NextDouble();
		}

		testingY[i - SIZE_OF_TRAINING_DATA] = new double[1];
		testingY[i - SIZE_OF_TRAINING_DATA][0] = testingX[i - SIZE_OF_TRAINING_DATA].Sum() < NUMBER_OF_INPUTS / 2.0 ? 0 : 1;
	}

}

var nn = new NeuralNetwork(NUMBER_OF_INPUTS, .001, 1, new BinaryCrossEntropyLossFunction());
nn.AddLayer(new DenseLayer(128, new ReluActivationFunction(.01)));
nn.AddLayer(new DenseLayer(1, new SigmoidActivationFunction()));

nn.Compile(new AdamOptimizer());

nn.Train(1500, trainingX, trainingY, 1, 10);

nn.Save("test.json");

var nn2 = NeuralNetwork.Load("test.json");
var accuracy = nn2.Test(testingX, testingY);

Console.WriteLine($"Test Accuracy: {accuracy}");


//var inputs = new double[4][]
//{
//	[-1, -1],
//	[-1, 1],
//	[1, -1],
//	[1, 1],
//};

//var outputs = new double[4][]
//{
//	[-1],
//	[1],
//	[1],
//	[-1],
//};

//var nn = new NeuralNetwork(2, .1, new MeanSquaredErrorLossFunction());
//nn.AddLayer(new DenseLayer(6, new TanHActivationFunction()));
//nn.AddLayer(new DenseLayer(1, new TanHActivationFunction()));

//nn.Train(5000, inputs, outputs, 2, true);

//nn.Save("test.json");

//var nn2 = NeuralNetwork.Load("test.json");

//Console.WriteLine(nn2.Test(inputs, outputs));