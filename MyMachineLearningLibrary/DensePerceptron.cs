using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class DensePerceptron : IPerceptron
	{
		public double Output { get; set; }
		public double[] Weights { get; set; }
		public double Bias { get; set; }
		public IActivationFunction ActivationFunction { get; set; }

		private void RandomizeWeightsAndBias()
		{
			Random rng = new();
			for (int i = 0; i < Weights.Length; i++)
			{
				Weights[i] = (rng.NextDouble() * 2) - 1;
			}
			Bias = (rng.NextDouble() * 2) - 1;
		}

		public DensePerceptron()
		{
			Weights = Array.Empty<double>();
			ActivationFunction = new NotDefinedActivationFunction();
		}

		public DensePerceptron(int numberOfWeights, IActivationFunction ActivationFunction)
		{
			Weights = new double[numberOfWeights];
			RandomizeWeightsAndBias();
			this.ActivationFunction = ActivationFunction;
		}

		public double Activate(double[] inputs)
		{
			Output = Bias;

			for (int i = 0; i < inputs.Length; i++)
			{
				Output += inputs[i] * Weights[i];
			}

			Output = ActivationFunction.ActivateFunction(Output);

			return Output;
		}
	}
}
