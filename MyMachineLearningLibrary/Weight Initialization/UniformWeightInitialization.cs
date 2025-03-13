using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary.Weight_Initialization
{
	public class UniformWeightInitialization : IWeightInitializtion
	{
		public double Min { get; set; }
		public double Max { get; set; }
		public UniformWeightInitialization(double min = -1, double max = 1)
		{
			Min = min;
			Max = max;
		}
		public void InitializeWeights(NeuralNetwork neuralNetwork)
		{
			//Ignore the input layer
			for (int i = 1; i < neuralNetwork.Layers.Count; i++)
			{
				var layer = neuralNetwork.Layers[i];
				foreach (var perceptron in layer.Perceptrons)
				{
					perceptron.UniformRandomizeWeights(Min, Max);
				}
			}
		}
	}
}
