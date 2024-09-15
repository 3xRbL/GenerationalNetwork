package main

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"gonum.org/v1/gonum/mat"
)

// Sigmoid activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// MeanSquaredError calculates the error between two vectors (actual and predicted)
func meanSquaredError(predicted, actual *mat.Dense) float64 {
	r, c := predicted.Dims()
	errorSum := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			diff := predicted.At(i, j) - actual.At(i, j)
			errorSum += diff * diff
		}
	}
	return errorSum / float64(r*c)
}

// NeuralNetwork represents a simple feedforward neural network
type NeuralNetwork struct {
	inputLayerSize   int
	hiddenLayerSizes []int
	outputLayerSize  int
	weights          []*mat.Dense
}

// NewNetwork initializes a new neural network with random weights for multiple layers
func NewNetwork(inputSize int, hiddenSizes []int, outputSize int) *NeuralNetwork {
	rand.Seed(time.Now().UnixNano())

	// Collect the full architecture sizes including input, hidden, and output layers
	layerSizes := append([]int{inputSize}, hiddenSizes...)
	layerSizes = append(layerSizes, outputSize)

	weights := make([]*mat.Dense, len(layerSizes)-1)

	// Initialize weight matrices for each layer transition
	for i := 0; i < len(layerSizes)-1; i++ {
		weights[i] = mat.NewDense(layerSizes[i], layerSizes[i+1], nil)
		for r := 0; r < layerSizes[i]; r++ {
			for c := 0; c < layerSizes[i+1]; c++ {
				weights[i].Set(r, c, rand.Float64()*2-1) // Random weights between -1 and 1
			}
		}
	}

	return &NeuralNetwork{
		inputLayerSize:   inputSize,
		hiddenLayerSizes: hiddenSizes,
		outputLayerSize:  outputSize,
		weights:          weights,
	}
}

// Forward propagates the inputs through the network with multiple hidden layers
func (nn *NeuralNetwork) Forward(X *mat.Dense) *mat.Dense {
	input := X

	// Pass through all hidden layers
	for i := 0; i < len(nn.weights)-1; i++ {
		var z mat.Dense
		z.Mul(input, nn.weights[i])          // z = input * weights
		input = applyActivation(&z, sigmoid) // Apply activation
	}

	// Last layer (output layer)
	var zFinal mat.Dense
	zFinal.Mul(input, nn.weights[len(nn.weights)-1]) // Output layer
	output := applyActivation(&zFinal, sigmoid)

	return output
}

// Helper function to apply an activation function to a matrix
func applyActivation(m *mat.Dense, activation func(float64) float64) *mat.Dense {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, activation(m.At(i, j)))
		}
	}
	return result
}

// Create population of neural networks
func createPopulation(popSize, inputSize, outputSize int, hiddenSizes []int) []*NeuralNetwork {
	population := make([]*NeuralNetwork, popSize)
	for i := range population {
		population[i] = NewNetwork(inputSize, hiddenSizes, outputSize)
	}
	return population
}

// cloneNetwork creates a deep copy of a neural network
func cloneNetwork(nn *NeuralNetwork) *NeuralNetwork {
	cloned := &NeuralNetwork{
		inputLayerSize:   nn.inputLayerSize,
		hiddenLayerSizes: append([]int{}, nn.hiddenLayerSizes...), // Copy hidden layer sizes
		outputLayerSize:  nn.outputLayerSize,
		weights:          make([]*mat.Dense, len(nn.weights)),
	}
	for i, w := range nn.weights {
		r, c := w.Dims()
		weightsCopy := mat.NewDense(r, c, nil)
		weightsCopy.Copy(w)
		cloned.weights[i] = weightsCopy
	}
	return cloned
}

// Mutate the target network by cloning the best network and applying mutations
func mutate(bestNetwork *NeuralNetwork, targetNetwork *NeuralNetwork, mutationRate float64) {
	// Clone the best network
	clonedNetwork := cloneNetwork(bestNetwork)

	// Apply mutations to the cloned network
	for _, weightMatrix := range clonedNetwork.weights {
		r, c := weightMatrix.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				if rand.Float64() < mutationRate {
					weightMatrix.Set(i, j, weightMatrix.At(i, j)+mutationRate*(rand.Float64()*2-1))
				}
			}
		}
	}

	// Replace target network with the mutated cloned network
	*targetNetwork = *clonedNetwork
}

// Evaluate fitness of the neural network by comparing its prediction with the actual next letter
func evaluate(nn *NeuralNetwork, input *mat.Dense, correctNextLetter *mat.Dense) float64 {
	// Forward pass to get the predicted next letter
	predicted := nn.Forward(input)

	// Calculate mean squared error between predicted and actual next letter (as one-hot vector)
	return -meanSquaredError(predicted, correctNextLetter) // Negative error since higher fitness is better
}

// Evolve population: select top performers and mutate them
func evolve(population []*NeuralNetwork, inputs, correctOutputs []*mat.Dense, mutationRate float64) {
	// Evaluate fitness of each network
	fitnesses := make([]float64, len(population))
	for i, nn := range population {
		fitness := 0.0
		for j, input := range inputs {
			fitness += evaluate(nn, input, correctOutputs[j])
		}
		fitnesses[i] = fitness
	}

	// Sort population by fitness
	sortByFitness(population, fitnesses)

	// Mutate the top 50% of the population
	for i := len(population)/2 + 1; i < len(population)-1; i++ {
		mutate(population[0], population[i], mutationRate)
	}

	population[len(population)-1] = NewNetwork(population[0].inputLayerSize, population[0].hiddenLayerSizes, population[0].outputLayerSize)
}

// Helper function to sort the population by fitness
func sortByFitness(population []*NeuralNetwork, fitnesses []float64) {
	for i := 0; i < len(fitnesses)-1; i++ {
		for j := 0; j < len(fitnesses)-i-1; j++ {
			if fitnesses[j] < fitnesses[j+1] {
				// Swap fitnesses
				fitnesses[j], fitnesses[j+1] = fitnesses[j+1], fitnesses[j]
				// Swap networks
				population[j], population[j+1] = population[j+1], population[j]
			}
		}
	}
}

// Save the network to a binary file
func (nn *NeuralNetwork) Save(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(nn)
}

// Load the network from a binary file
func LoadNetwork(filename string) (*NeuralNetwork, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var nn NeuralNetwork
	err = decoder.Decode(&nn)
	if err != nil {
		return nil, err
	}
	return &nn, nil
}

// Adjust the population size by adding or removing networks, using the first network to infer sizes
func adjustPopulationSize(population []*NeuralNetwork, newSize int) []*NeuralNetwork {
	if len(population) == 0 {
		fmt.Println("Error: Population is empty, cannot adjust size.")
		return population
	}

	// Use the first network to infer input size, output size, and hidden sizes
	exampleNetwork := population[0]
	inputSize := exampleNetwork.inputLayerSize
	outputSize := exampleNetwork.outputLayerSize
	hiddenSizes := exampleNetwork.hiddenLayerSizes

	currentSize := len(population)

	// If the new size is larger, append new networks
	if newSize > currentSize {
		for i := currentSize; i < newSize; i++ {
			newNetwork := NewNetwork(inputSize, hiddenSizes, outputSize)
			population = append(population, newNetwork)
		}
	} else if newSize < currentSize {
		// If the new size is smaller, truncate the slice
		population = population[:newSize]
	}

	return population
}

// Test and show the output of the top-performing network
func testAndShowResults(nn *NeuralNetwork, inputs []*mat.Dense, correctOutputs []*mat.Dense) {
	fmt.Println("Testing top-performing network on given inputs:")
	for i, input := range inputs {
		fmt.Printf("Test case %d:\n", i+1)

		// Get prediction and actual next letter (as one-hot)
		predicted := nn.Forward(input)
		fmt.Println("Predicted:", mat.Formatted(predicted, mat.Prefix(" "), mat.Squeeze()))
		fmt.Println("Actual:", mat.Formatted(correctOutputs[i], mat.Prefix(" "), mat.Squeeze()))

		// Calculate and display mean squared error
		errorValue := meanSquaredError(predicted, correctOutputs[i])
		fmt.Printf("Mean Squared Error: %.6f\n\n", errorValue)
	}
}

func main() {
	// Example usage
	inputSize := 5
	outputSize := 5
	hiddenSizes := []int{5, 5}
	populationSize := 100
	mutationRate := 0.1
	generations := 1000

	// Create initial population of neural networks
	population := createPopulation(populationSize, inputSize, outputSize, hiddenSizes)

	// Example inputs (e.g., encoded letters or words as one-hot vectors)
	inputs := []*mat.Dense{
		mat.NewDense(1, inputSize, []float64{1, 0, 0, 0, 0}), // A
		mat.NewDense(1, inputSize, []float64{0, 1, 0, 0, 0}), // B
	}

	// Correct next letter outputs (one-hot encoded)
	correctOutputs := []*mat.Dense{
		mat.NewDense(1, outputSize, []float64{0, 1, 0, 0, 0}), // B
		mat.NewDense(1, outputSize, []float64{0, 0, 1, 0, 0}), // C
	}

	// Evolve population over generations
	for g := 0; g < generations; g++ {
		evolve(population, inputs, correctOutputs, mutationRate)
		fmt.Printf("Generation %d complete\n", g+1)
	}

	// Test and show results of the top-performing network
	testAndShowResults(population[0], inputs, correctOutputs)

	// Save the top-performing network to a file
	//err := population[0].Save("best_network.gob")
	//if err != nil {
	//	fmt.Println("Error saving network:", err)
	//} else {
	//	fmt.Println("Best network saved to best_network.gob")
	//}
}
