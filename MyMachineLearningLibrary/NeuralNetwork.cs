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
		
		public NeuralNetwork()
		{
			Layers = new List<ILayer>();
		}

		public NeuralNetwork(int NumberOfInputs, double LearningRate) 
		{
			Layers = [];
			this.LearningRate = LearningRate;
			this.NumberOfInputs = NumberOfInputs;
		}

		public void AddLayer(ILayer layer)
		{
			if (Layers.Count == 0)
			{
				layer.InitializeLayer(NumberOfInputs, new NeuralNetMatrix(NumberOfInputs, 1));
			}
			else
			{
				var previousLayer = Layers.Last();
				layer.InitializeLayer(previousLayer.NumberOfPerceptrons, previousLayer.Gradients);
			}
			
			Layers.Add(layer);
		}

		public double[] Predict(double[] inputsArray)
		{
			var result = new NeuralNetMatrix(inputsArray);

			for (int i = 0; i < Layers.Count; i++)
			{
				var layer = Layers[i];
				result = layer.FeedForward(result);
			}

			return result.Flatten();
		}

		public void Train(int numberOfEpochs, double[][] trainInputsArray, double[][] trainTargetsArray, int batchSize = 1 /*Stochastic Gradient Descent by default*/)
		{
			for (int currentEpoch = 1; currentEpoch <= numberOfEpochs; currentEpoch++)
			{
				int batchCount = 0;

				var shuffledIndexes = Shuffle(trainInputsArray.Length);

				for (int i = 0; i < trainInputsArray.Length; i++)
				{
					//Shuffle the indexes so that the inputs are given in a random order
					int k = shuffledIndexes[i];
					var inputsArray = trainInputsArray[k];
					var targetsArray = trainTargetsArray[k];

					Predict(inputsArray);
					var outputMatrix = Layers.Last().LayerOutputs;

					// Calculate the output errors
					var currentErrors = new NeuralNetMatrix(targetsArray);
					currentErrors.Subtract(outputMatrix);

					// Backpropagate the rrors
					for (int j = Layers.Count - 1; j > 0; j--)
					{
						var layer = (ILayer)Layers[j];
						var weights = layer.Weights;
						var biases = layer.Biases;
						var layerOutputs = layer.LayerOutputs;
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

						// Calculate Error
						var weightsTransposed = NeuralNetMatrix.Transpose(weights);
						currentErrors = NeuralNetMatrix.DotProduct(weightsTransposed, currentErrors);
					}
				}
			}
		}

		public void Save(string filePath)
		{
			//var json = JsonSerializer.Serialize(Layers);
			//File.WriteAllText(filePath, json);
			var json = JsonSerializer.Serialize(this);
			File.WriteAllText(filePath, json);
		}

		public static NeuralNetwork Load(string filePath) 
		{
			//var json = File.ReadAllText(filePath);
			//var layers = JsonSerializer.Deserialize<List<ILayer>>(json);
			//var neuralNetwork = new NeuralNetwork();
			//neuralNetwork.Layers = layers;
			//return neuralNetwork;
			var json = File.ReadAllText(filePath);
			return JsonSerializer.Deserialize<NeuralNetwork>(json);
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
