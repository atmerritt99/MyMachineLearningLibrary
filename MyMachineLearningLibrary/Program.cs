// See https://aka.ms/new-console-template for more information
using MyMachineLearningLibrary;


/*
 
 TODO:
	Rename variables, write better comments
	Currently, Save and Load works by making all variables public methods, I should look into changing this
	Add Softmax Activation and Categorical Cross Entropy Loss Function
 */


//var nn = new NeuralNetwork(2, .1, new BinaryCrossEntropyLossFunction());
//nn.AddLayer(new DenseLayer(4, new SigmoidActivationFunction()));
//nn.AddLayer(new DenseLayer(1, new SigmoidActivationFunction()));

var nn = NeuralNetwork.Load("test.json");

// XOR Problem for testing
var x = new double[][]
{
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1],
};

var y = new double[][]
{
	[0],
	[1],
	[1],
	[0],
};

//nn.Train(100000, x, y, 1, true);

foreach (var input in x)
{
	Console.WriteLine(nn.Predict(input)[0]);
}

//nn.Save("test.json");