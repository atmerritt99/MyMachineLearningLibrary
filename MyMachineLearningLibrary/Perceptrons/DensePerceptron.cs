using MyMachineLearningLibrary.Activation_Functions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary.Perceptrons
{
	public class DensePerceptron : IPerceptron
	{
		public double Output { get; set; }
		public double[] Weights { get; set; }
		public double Bias { get; set; }
		public IActivationFunction ActivationFunction { get; set; }

		public void UniformRandomizeBias(double min = -1, double max = 1)
		{
			RandomExtension rng = new();
			Bias = rng.NextDouble(min, max);
		}

		public void NormalRandomizeBias(double mean = 0, double standardDeviation = .1)
		{
			RandomExtension rng = new();
			Bias = rng.NormalDistribution(mean, standardDeviation);
		}

		public void NormalRandomizeWeights(double mean = 0, double standardDeviation = .1)
		{
			RandomExtension rng = new();
			for (int i = 0; i < Weights.Length; i++)
			{
				Weights[i] = rng.NormalDistribution(mean, standardDeviation);
			}
		}

		public void UniformRandomizeWeights(double min = -1, double max = 1)
		{
			RandomExtension rng = new();
			for (int i = 0; i < Weights.Length; i++)
			{
				Weights[i] = rng.NextDouble(min, max);
			}
		}

		public DensePerceptron()
		{
			Weights = Array.Empty<double>();
			ActivationFunction = new NotDefinedActivationFunction();
		}

		public DensePerceptron(int numberOfWeights, IActivationFunction ActivationFunction)
		{
			Weights = new double[numberOfWeights];
			this.ActivationFunction = ActivationFunction;
		}

		public double WeightedSum(double[] inputs)
		{
			Output = Bias;

			for (int i = 0; i < inputs.Length; i++)
			{
				Output += inputs[i] * Weights[i];
			}

			return Output;
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
