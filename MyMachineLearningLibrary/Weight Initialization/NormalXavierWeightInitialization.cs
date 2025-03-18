using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary.Weight_Initialization
{
	public class NormalXavierWeightInitialization : IWeightInitializtion
	{
		public void InitializeWeights(NeuralNetwork neuralNetwork)
		{
			//Ignore the input layer
			for (int i = 1; i < neuralNetwork.Layers.Count; i++)
			{
				var layer = neuralNetwork.Layers[i];
				foreach (var perceptron in layer.Perceptrons)
				{
					double numberOfInputs = neuralNetwork.Layers[i - 1].NumberOfPerceptrons;
					double numberOfOutputs = 0;

					if (i < neuralNetwork.Layers.Count - 1)
					{
						numberOfOutputs = neuralNetwork.Layers[i + 1].NumberOfPerceptrons;
					}

					double x = Math.Sqrt(2.0 / (numberOfInputs + numberOfOutputs));

					perceptron.UniformRandomizeWeights(0, x);
				}
			}
		}
	}
}
