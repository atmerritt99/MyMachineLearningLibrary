using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class NormalWeightInitialization : IWeightInitializtion
	{
		public double Mean { get; set; }
		public double StandardDeviation { get; set; }
		public NormalWeightInitialization(double mean = 0, double standardDeviation = .1)
		{
			Mean = mean;
			StandardDeviation = standardDeviation;
		}
		public void InitializeWeights(NeuralNetwork neuralNetwork)
		{
			//Ignore the input layer
			for (int i = 1; i < neuralNetwork.Layers.Count; i++)
			{
				var layer = neuralNetwork.Layers[i];
				foreach (var perceptron in layer.Perceptrons)
				{
					perceptron.NormalRandomizeWeights(Mean, StandardDeviation);
				}
			}
		}
	}
}
