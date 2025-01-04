using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace MyMachineLearningLibrary
{
	[JsonInterfaceConverter(typeof(InterfaceConverter<IPerceptron>))]
	public interface IPerceptron
	{
		public double Output { get; set; }
		public double[] Weights { get; set; }
		public double Bias { get; set; }
		public double Activate(double[] inputs);
	}
}
