// ************************************************************************************************
// <copyright file="NeuralNetwork.cs">
//   Copyright (c) 2017 Thomas Weller
// </copyright>
// <authors>
//   <author>Thomas Weller</author>
// </authors>
// <summary>
// C# version of the code in 'Make Your Own Neural Network'.
// Mimicks the code from pp. 169, implementing a simple console program.
// </summary>
// ************************************************************************************************

// Standard libraries
using System;

// Additional math libraries
using Accord.Math;
using Accord.Statistics.Distributions.Univariate;


namespace MakeYourOwnNeuralNetworkUsingCSharp
{
    public class NeuralNetwork
    {
        public readonly int InputNodes;
        public readonly int HiddenNodes;
        public readonly int OutputNodes;

        public readonly double LearningRate;

        public readonly Func<double, double> ActivationFunction;

        // link weight matrices, wih and who
        // weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        // w11 w21
        // w12 w22 etc
        public double[,] Wih;
        public double[,] Who;

        // initialise the neural network
        public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate)
        {
            // set number of nodes in each input, hidden, output layer
            HiddenNodes = hiddenNodes;
            InputNodes = inputNodes;
            OutputNodes = outputNodes;

            // Create weigth matrices
            Wih = new double[HiddenNodes, InputNodes];
            Who = new double[OutputNodes, HiddenNodes];

            // Initialize the weight matrices.
            // Sample their start values from a normal probability distribution centred around zero
            // and with a standard deviation that is related to the number of incoming links into a
            // node, 1 /√(number of incoming links).
            NormalDistribution distribution = new NormalDistribution(0.0, 1.0 / Math.Sqrt(HiddenNodes));

            for (int i = 0; i < Wih.Rows(); i++)
            {
                Wih.SetRow(i, distribution.Generate(InputNodes));
            }

            distribution = new NormalDistribution(0.0, 1.0 / Math.Sqrt(OutputNodes));

            for (int i = 0; i < Who.Rows(); i++)
            {
                Who.SetRow(i, distribution.Generate(HiddenNodes));
            }

            // learning rate
            LearningRate = learningRate;

            // activation function is the sigmoid function
            ActivationFunction = Sigmoid;
        }

        // query the neural network
        public double[] Query(double[] inputsList)
        {
            // Convert inputs list (one-dimensional) to two-dimensional, one-column matrix.
            double[,] inputs = Matrix.Create<double>(inputsList.Length, 1);
            inputs.SetColumn(0, inputsList);

            // calculate signals into hidden layer
            double[,] hiddenInputs = Wih.Dot(inputs);
            // calculate the signals emerging from hidden layer
            double[,] hiddenOutputs = hiddenInputs.Apply(x => ActivationFunction(x));

            // calculate signals into final output layer
            double[,] finalInputs = Who.Dot(hiddenOutputs);
            // calculate the signals emerging from final output layer
            double[,] finalOutputs = finalInputs.Apply(x => ActivationFunction(x));

            // Convert output node matrix back to a 1-dimensional list of values.
            return finalOutputs.GetColumn(0);
        }

        // train the neural network
        public void Train(double[] inputsList, double[] targetsList)
        {
            // Convert inputs and targets list (one-dimensional) to two-dimensional, one-column matrices.
            double[,] inputs = new double[inputsList.Length, 1];
            inputs.SetColumn(0, inputsList);

            // calculate signals into hidden layer
            double[,] hiddenInputs = Wih.Dot(inputs);
            // calculate the signals emerging from hidden layer
            double[,] hiddenOutputs = hiddenInputs.Apply(x => ActivationFunction(x));

            // calculate signals into final output layer
            double[,] finalInputs = Who.Dot(hiddenOutputs);
            // calculate the signals emerging from final output layer
            double[,] finalOutputs = finalInputs.Apply(x => ActivationFunction(x));

            // output layer error is the (target - actual)
            double[] outputError = targetsList.Subtract(finalOutputs.GetColumn(0));
            // hidden layer error is the output_errors, split by weights, recombined at hidden nodes
            double[] hiddenError = Who.TransposeAndDot(outputError);

            // update the weights for the links between the hidden and output layers
            UpdateWeigths(hiddenOutputs, finalOutputs, ref Who, outputError);

            // update the weights for the links between the input and hidden layers
            UpdateWeigths(inputs, hiddenOutputs, ref Wih, hiddenError);
        }

        // ----------------------------------------------------------------------------------------

        // Helper function for updating weight values during backpropagation.
        private void UpdateWeigths(double[,] layer1, double[,] layer2, ref double[,] weigths, double[] errorList)
        {
            // Convert errors list (one-dimensional) to two-dimensional, one-column matrix.
            double[,] errors = Matrix.Create<double>(errorList.Length, 1);
            errors.SetColumn(0, errorList);

            // Calculate delta matrix (differences for updating weight values).
            double[,] deltas = Elementwise.Multiply(errors, layer2.Apply(x => x * (1 - x)))
                                          .Dot(layer1.Transpose())
                                          .Apply(x => x * .1);

            // Apply delta matrix.
            weigths = weigths.Add(deltas);
        }

        // ----------------------------------------------------------------------------------------

        // Sigmoid function for node activation. 
        private double Sigmoid(double x)
        {
            return 1/(1 + Math.Exp(-1 * x));
        }
    }
}