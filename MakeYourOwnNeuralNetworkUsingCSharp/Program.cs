// ************************************************************************************************
// <copyright file="Program.cs">
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
using System.Collections.Generic;
using System.IO;
using System.Linq;

// Additional math libraries
using Accord.Math;

namespace MakeYourOwnNeuralNetworkUsingCSharp
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Console.Write("Press Enter to start..."); 
            Console.ReadLine();

            int counter = 0;

            // number of input, hidden and output nodes 
            int inputNodes = 784;
            int hiddenNodes = 200;
            int outputNodes = 10;

            // learning rate
            double learningRate = 0.1;

            // create instance of neural network
            NeuralNetwork n = new NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate);

            // load the mnist training data CSV file into a list
            string[] trainingDataList = File.ReadAllLines("../../mnist_dataset/mnist_train.csv");

            // train the neural network
            // ************************
            Console.WriteLine("Train the neural network...");

            // epochs is the number of times the training data set is used for training
            int epochs = 5;

            for (int i = 0; i < epochs; i++)
            {
                Console.Write("Epoch " + (i + 1) + " ");

                // go through all records in the training data set
                foreach (string record in trainingDataList)
                {
                    // split the record by the ',' commas
                    string[] allValues = record.Split(',');
                    // scale and shift the inputs
                    double[] inputs = allValues.Skip(1)
                        .Select(b => Convert.ToByte(b)/255.0*0.99 + 0.01)
                        .ToArray();

                    // create the target output values (all 0.01, except the desired label which is 0.99)
                    double[] targets = {.01, .01, .01, .01, .01, .01, .01, .01, .01, .01};
                    // allValues[0] is the target label for this record
                    targets[Convert.ToByte(allValues[0])] = 0.99;

                    counter++;
                    if (counter % 600 == 0)
                    {
                        Console.Write(".");
                    }

                    n.Train(inputs, targets);
                }

                Console.WriteLine();
            }

            // load the mnist test data CSV file into a list
            string[] testDataList = File.ReadAllLines("../../mnist_dataset/mnist_test.csv");

            // test the neural network
            // ***********************
            Console.WriteLine("Test the neural network...");
            counter = 0;

            // scorecard for how well the network performs, initially empty
            List<byte> scorecard = new List<byte>();

            counter = 0;

            // go through all the records in the test data set
            foreach (string record in testDataList)
            {
                counter++;

                // split the record by the ',' commas
                string[] allValues = record.Split(',');

                // correct answer is first value
                int correctLabel = Convert.ToInt32(allValues[0]);

                // scale and shift the inputs
                double[] inputs = allValues.Skip(1)
                    .Select(b => Convert.ToByte(b)/255.0*0.99 + 0.01)
                    .ToArray();

                // query the network
                double[] outputs = n.Query(inputs);

                // the index of the highest value corresponds to the label
                int label = outputs.IndexOf(outputs.Max());

                // network's answer matches correct answer, add 1 to scorecard
                if (label == correctLabel)
                {
                    scorecard.Add(1);
                }
                // network's answer doesn't match correct answer, add 0 to scorecard
                else
                {
                    scorecard.Add(0);
                    Console.WriteLine(
                        "Training mismatch at #" + counter + ": Label was " + correctLabel + 
                        ", but the Neural Net classified the data as " + label + "}.");
                }
            }

            // calculate the performance score, the fraction of correct answers
            Console.WriteLine("  Performance = " + scorecard.Average(b => b));

            Console.Write("Press Enter to quit...");
            Console.ReadLine();
        }
    }
}
