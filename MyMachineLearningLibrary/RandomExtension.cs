using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	public class RandomExtension : Random
	{
		public RandomExtension() : base()
		{
			
		}

		public double NextDouble(double min, double max)
		{
			return (NextDouble() * (max - min)) + min;
		}

		public double NormalDistribution(double mean, double stdDev)
		{
			double u1 = 1.0 - NextDouble(); //uniform(0,1] random doubles
			double u2 = 1.0 - NextDouble();
			double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
						 Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
			double randNormal = mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)
			return randNormal;
		}
	}
}
