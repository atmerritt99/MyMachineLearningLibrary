﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class MeanSquaredErrorLossFunction : ILossFunction
	{
		public double CalculateLoss(NeuralNetMatrix targets, NeuralNetMatrix outputs, out NeuralNetMatrix lossMatrix, out NeuralNetMatrix errorsDirections)
		{
			lossMatrix = NeuralNetMatrix.Subtract(targets, outputs);
			lossMatrix.Multiply(lossMatrix);

			errorsDirections = NeuralNetMatrix.Compare(targets, outputs);

			return lossMatrix.Average;
		}
	}
}