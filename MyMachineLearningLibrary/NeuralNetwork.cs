using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;

namespace MyMachineLearningLibrary
{
	public class NeuralNetwork
	{
		public List<ILayer> Layers { get; set; }
		public double LearningRate { get; set; }
		public int NumberOfInputs { get; set; }
		public ILossFunction LossFunction { get; set; }

		private NeuralNetwork()
		{
			Layers = new List<ILayer>();
			LossFunction = new NotDefinedLossFunction();
		}

		public NeuralNetwork(int NumberOfInputs, double LearningRate, ILossFunction LossFunction) 
		{
			Layers = [new InputLayer(NumberOfInputs)];
			this.LearningRate = LearningRate;
			this.NumberOfInputs = NumberOfInputs;
			this.LossFunction = LossFunction;
		}

		public void AddLayer(ILayer layer)
		{
			var previousLayer = Layers.Last();
			layer.InitializeLayer(previousLayer.NumberOfPerceptrons, previousLayer.Gradients);
			Layers.Add(layer);
		}

		public double[] Predict(double[] inputsArray)
		{
			var result = new NeuralNetMatrix(inputsArray);

			foreach(var layer in Layers)
			{
				result = layer.FeedForward(result);
			}

			return result.Flatten();
		}

		public int[] Classify(double[] inputsArray)
		{
			var predictions = Predict(inputsArray);
			var result = new int[predictions.Length];

			if (predictions.Length == 1)
			{
				var prediction = predictions[0];

				result[0] = prediction >= .5 ? 1 : 0;

				return result;
			}

			int indexOfMaxValue = predictions.ToList().IndexOf(predictions.Max());
			result[indexOfMaxValue] = 1;

			return result;
		}

		public void Train(int numberOfEpochs, double[][] trainInputsArray, double[][] trainTargetsArray, int batchSize = 1 /*Stochastic Gradient Descent by default*/, bool verbose = false)
		{
			for (int currentEpoch = 1; currentEpoch <= numberOfEpochs; currentEpoch++)
			{
				int batchCount = 0;

				var shuffledIndexes = Shuffle(trainInputsArray.Length);

				double cost = 0; // The cost is the average of all the loss

				for (int i = 0; i < trainInputsArray.Length; i++)
				{
					//Shuffle the indexes so that the inputs are given in a random order
					int k = shuffledIndexes[i];
					var inputsArray = trainInputsArray[k];
					var targetsArray = trainTargetsArray[k];

					Predict(inputsArray);
					var outputsMatrix = Layers.Last().LayerOutputs;

					// Calculate the errors
					var targetsMatrix = new NeuralNetMatrix(targetsArray);
					var currentErrors = LossFunction.CalculateDerivativeOfLoss(targetsMatrix, outputsMatrix);
					cost += LossFunction.CalculateLoss(targetsMatrix, outputsMatrix);

					// Backpropagate the errors
					for (int j = Layers.Count - 1; j > 0; j--)
					{
						var layer = Layers[j];
						var previousLayerOutputs = Layers[j - 1].LayerOutputs;

						//Calculate the output gradients
						layer.CalculateGradients(currentErrors, LearningRate);

						//Apply Gradients
						if ((i + 1) % batchSize == 0 || (i + 1) == trainInputsArray.Length)
						{
							if ((i + 1) == trainInputsArray.Length)
							{
								int lengthFinalBatch = trainInputsArray.Length - (batchSize * batchCount);
								layer.ApplyGradients(previousLayerOutputs, lengthFinalBatch);
							}
							else
							{
								batchCount++;
								layer.ApplyGradients(previousLayerOutputs, batchSize);
							}
						}

						// Calculate this layer's Errors
						var weightsTransposed = layer.Weights.Transpose();
						currentErrors = NeuralNetMatrix.DotProduct(weightsTransposed, currentErrors);
					}
				}

				cost /= trainInputsArray.Length;

				if (verbose)
					Console.WriteLine($"EPOCH: {currentEpoch}\tCOST: {cost}");
			}
		}

		public double Test(double[][] testInputsArray, double[][] testTargetsArray)
		{
			var classifications = new int[testInputsArray.Length][];
			for(int i = 0; i < testInputsArray.Length; i++)
			{
				classifications[i] = Classify(testInputsArray[i]);
			}

			double accuracy = 1;
			for (int i = 0; i < classifications.Length; i++)
			{
				int predictedClass = classifications[i].ToList().IndexOf(1);
				int actualClass = testTargetsArray[i].ToList().IndexOf(1);

				if (predictedClass != actualClass)
				{
					accuracy -= (1.0 / testInputsArray.Length);
				}
			}

			return accuracy;
		}

		public void Save(string filePath)
		{
			var json = JsonSerializer.Serialize(this);
			File.WriteAllText(filePath, json);
		}

		public static NeuralNetwork Load(string filePath) 
		{
			var json = File.ReadAllText(filePath);
			return JsonSerializer.Deserialize<NeuralNetwork>(json) ?? new NeuralNetwork();
		}

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
