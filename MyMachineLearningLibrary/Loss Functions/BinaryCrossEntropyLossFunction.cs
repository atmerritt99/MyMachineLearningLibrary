namespace MyMachineLearningLibrary.Loss_Functions
{
	public class BinaryCrossEntropyLossFunction : ILossFunction
	{
		public MatrixExtension CalculateDerivativeOfLoss(MatrixExtension targets, MatrixExtension outputs)
		{
			var lossMatrix = outputs.Subtract(targets);
			var oneMinusOutputs = MatrixExtension.Subtract(1, outputs);
			var x = outputs.Multiply(oneMinusOutputs);
			lossMatrix = lossMatrix.Divide(x);
			return lossMatrix;
		}

		public double CalculateLoss(MatrixExtension targets, MatrixExtension outputs)
		{
			var lossMatrix = targets.Copy();
			var logOutputs = outputs.Log();
			lossMatrix = lossMatrix.Multiply(logOutputs);

			var oneMinusTargets = MatrixExtension.Subtract(1, targets);
			var oneMinusOutputs = MatrixExtension.Subtract(1, outputs);
			var logOneMinusOutputs = oneMinusOutputs.Log();
			var y = oneMinusTargets.Multiply(logOneMinusOutputs);

			lossMatrix = lossMatrix.Add(y);

			lossMatrix = lossMatrix.Multiply(-1);

			return lossMatrix.RowSumOfAvg;
		}
	}
}
