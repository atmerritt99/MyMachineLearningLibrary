using MyMachineLearningLibrary.Activation_Functions;
using MyMachineLearningLibrary.Layers;
using MyMachineLearningLibrary.Loss_Functions;
using MyMachineLearningLibrary.Optimizers;
using MyMachineLearningLibrary.Weight_Initialization;
using System.Text.Json;

namespace MyMachineLearningLibrary
{
	public class NeuralNetwork
	{
		public IOptimizer Optimizer { get; set; }
		public List<ILayer> Layers { get; set; }
		public double LearningRate { get; set; }
		public int DecayRate { get; set; }
		public int NumberOfInputs { get; set; }
		public ILossFunction LossFunction { get; set; }
		public IWeightInitializtion WeightInitializtion { get; set; }
		private NeuralNetwork()
		{
			Layers = new List<ILayer>();
			LossFunction = new NotDefinedLossFunction();
			Optimizer = new GradientDescentOptimizer();
			WeightInitializtion = new UniformXavierWeightInitialization();
		}

		public NeuralNetwork(int NumberOfInputs, double LearningRate, int DecayRate, ILossFunction LossFunction, IWeightInitializtion WeightInitializtion, IOptimizer Optimizer) 
		{
			Layers = [];
			this.LearningRate = LearningRate;
			this.NumberOfInputs = NumberOfInputs;
			this.LossFunction = LossFunction;
			this.DecayRate = DecayRate;
			this.Optimizer = Optimizer;
			this.WeightInitializtion = WeightInitializtion;
		}

		private void Compile()
		{
			WeightInitializtion.InitializeWeights(this);
			Optimizer.Compile(this);
		}

		public void AddLayer(ILayer layer)
		{
			if(layer.ActivationFunction.GetType() == typeof(SoftmaxActivationFunction) && LossFunction.GetType() != typeof(CategoricalCrossEntropyLossFunction))
			{
				throw new Exception("Softmax Activation can only be used with Categorical Cross Entropy in this Library");
			}

			if(Layers.Count > 0)
			{
				var previousLayer = Layers.Last();
				layer.InitializeLayer(previousLayer.NumberOfPerceptrons, Layers.Count);
			}
			else
			{
				layer.InitializeLayer(NumberOfInputs, Layers.Count);
			}
			
			Layers.Add(layer);
		}

		/// <summary>
		/// Feeds the input array through the neural network
		/// </summary>
		/// <param name="inputsArray"></param>
		/// <returns>The output of the neural network</returns>
		public double[] Predict(double[] inputsArray)
		{
			var result = new MatrixExtension(inputsArray);

			foreach(var layer in Layers)
			{
				result = layer.FeedForward(result, false);
			}

			return result.Flatten();
		}

		private MatrixExtension Predict(MatrixExtension inputsMatrix)
		{
			var result = inputsMatrix.Copy();

			foreach (var layer in Layers)
			{
				result = layer.FeedForward(result);
				result = result.Transpose();
			}

			return result;
		}

		private int[] MakeClassification(double[] predictions)
		{
			var result = new int[predictions.Length];

			if (predictions.Length == 1)
			{
				var prediction = predictions[0];

				var t = Layers.Last().ActivationFunction.MinClass == 0 ? .5 : 0;

				result[0] = prediction >= t ? Layers.Last().ActivationFunction.MaxClass : Layers.Last().ActivationFunction.MinClass;

				return result;
			}

			int indexOfMaxValue = predictions.ToList().IndexOf(predictions.Max());
			result[indexOfMaxValue] = 1;

			for (int i = 0; i < result.Length; i++)
			{
				if (i != indexOfMaxValue)
				{
					result[i] = Layers.Last().ActivationFunction.MinClass;
				}
			}

			return result;
		}

		/// <summary>
		/// Given an input to the neural network, return the predicted class
		/// </summary>
		/// <param name="inputsArray"></param>
		/// <returns>If the output layer has only 1 perceptron, return an int array size 1 with either a 1 or a 0. Otherwise, return an int array with a 1 marking the predicted class, and a 0 marking the other classes</returns>
		public int[] Classify(double[] inputsArray)
		{
			var predictions = Predict(inputsArray);
			return MakeClassification(predictions);
		}

		public void Train(int numberOfEpochs, double[][] trainInputsArrays, double[][] trainTargetsArrays, int batchSize = 1 /*Stochastic Gradient Descent by default*/, int reportingRate = 1)
		{
			Compile();

			//Prepare Data
			var trainInputsMatrix = new MatrixExtension(trainInputsArrays);
			var trainTargetsMatrix = new MatrixExtension(trainTargetsArrays);

			//Create Batches
			var trainInputsBatches = new List<MatrixExtension>();
			var trainTargetsBatches = new List<MatrixExtension>();

			for (int i = 0; i < trainInputsMatrix.RowLength; i += batchSize)
			{
				int numberOfRows = batchSize;

				if (i + batchSize >= trainInputsMatrix.RowLength)
				{
					numberOfRows = trainInputsMatrix.RowLength - i;
				}

				trainInputsBatches.Add(trainInputsMatrix.GetRows(i, numberOfRows));
				trainTargetsBatches.Add(trainTargetsMatrix.GetRows(i, numberOfRows));
			}

			for (int currentEpoch = 1; currentEpoch <= numberOfEpochs; currentEpoch++)
			{
				var shuffledIndexes = Shuffle(trainInputsBatches.Count); //Shuffle the indexes so that the inputs are given in a random order
				double cost = 0; // The cost is the average of all the loss
				double accuracy = 0;

				foreach (var index in shuffledIndexes) 
				{
					var outputsMatrix = Predict(trainInputsBatches[index]);

					// Calculate the errors
					var targetsMatrix = trainTargetsBatches[index];
					var errors = LossFunction.CalculateDerivativeOfLoss(targetsMatrix, outputsMatrix).Transpose();
					cost += LossFunction.CalculateLoss(targetsMatrix, outputsMatrix);

					// Calculate the accuracy
					int counter = 0;
					foreach(var output in outputsMatrix.Values)
					{
						var classification = MakeClassification(output);
						accuracy = TestClassification(classification, targetsMatrix.Values[counter]) ? accuracy + 1 : accuracy;
						counter += 1;
					}

					// Backpropagate the errors
					for(int i = Layers.Count - 1; i >= 0; i--)
					{
						if(i > 0)
							errors = Layers[i].Backpropagate(errors, LearningRate, DecayRate, Layers[i - 1].Outputs, currentEpoch, Optimizer);
						else
							errors = Layers[i].Backpropagate(errors, LearningRate, DecayRate, trainInputsBatches[index].Transpose(), currentEpoch, Optimizer);
					}
				}

				cost /= trainInputsArrays.Length;
				accuracy /= trainInputsArrays.Length;

				if (reportingRate > 0 && currentEpoch % reportingRate == 0)
					Console.WriteLine($"EPOCH: {currentEpoch}\tCOST: {cost}\tTRAIN ACCURACY: {accuracy}");
			}
		}

		private bool TestClassification(int[] classifications, double[] targets)
		{
			for (int i = 0; i < classifications.Length; i++)
			{
				if (classifications[i] != targets[i])
					return false;
			}
			return true;
		}

		/// <summary>
		/// Tests the models ability to classify the data passed into this function
		/// </summary>
		/// <param name="testInputsArray"></param>
		/// <param name="testTargetsArray"></param>
		/// <returns>The accuracy of the model</returns>
		public double Test(double[][] testInputsArray, double[][] testTargetsArray)
		{
			var classifications = new int[testInputsArray.Length][];
			for(int i = 0; i < testInputsArray.Length; i++)
			{
				classifications[i] = Classify(testInputsArray[i]);
			}

			double accuracy = 0;
			for (int i = 0; i < classifications.Length; i++)
			{
				if(TestClassification(classifications[i], testTargetsArray[i]))
				{
					accuracy++;
				}
			}

			return accuracy / classifications.Length;
		}

		/// <summary>
		/// Serializes and saves the model to a JSON file
		/// </summary>
		/// <param name="filePath"></param>
		public void Save(string filePath)
		{
			var json = JsonSerializer.Serialize(this);
			File.WriteAllText(filePath, json);
		}

		/// <summary>
		/// Deserializes and loads the model from a JSON file
		/// </summary>
		/// <param name="filePath"></param>
		/// <returns>The deserialized model</returns>
		public static NeuralNetwork Load(string filePath) 
		{
			var json = File.ReadAllText(filePath);
			return JsonSerializer.Deserialize<NeuralNetwork>(json) ?? new NeuralNetwork();
		}

		/// <summary>
		/// This function fills an int array of size n with the numbers 0 - (n - 1) in a random order.
		/// This is used to feed the training inputs in a random order for better training outcomes
		/// </summary>
		/// <param name="max"></param>
		/// <returns></returns>
		private int[] Shuffle(int max)
		{
			int[] result = new int[max];

			for (int i = 0; i < max; i++)
			{
				result[i] = i;
			}

			int n = max;
			var rng = new Random();
			while (n > 1)
			{
				n--;
				int k = rng.Next(n + 1);
				int value = result[k];
				result[k] = result[n];
				result[n] = value;
			}

			return result;
		}
	}
}
