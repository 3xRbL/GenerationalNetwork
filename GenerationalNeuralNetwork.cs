public class GenerationNeuralNetwork
{
	private NeuralNetwork[] _neuralNetworks;
	private static Random _random;

	/// <summary>
	/// Initialize Generational Network with parameters
	/// </summary>
	/// <param name="networkCount">Neural network count inside</param>
	/// <param name="inputSize">Size of input of neural network</param>
	/// <param name="hiddenLayers">int array, member count is hiddenlayer count, and number is it's size</param>
	/// <param name="outputSize">Size of output of neural network</param>
	public GenerationNeuralNetwork(int networkCount, int inputSize, int[] hiddenLayers, int outputSize)
	{
		_neuralNetworks = new NeuralNetwork[networkCount];
		_random = new Random();

		for (var i = 0; i < networkCount; i++) _neuralNetworks[i] = new NeuralNetwork(inputSize, hiddenLayers, outputSize);
	}

	/// <summary>
	/// Saves Generational Network's parameters to a file
	/// </summary>
	/// <param name="fileName">Name of file to save to</param>
	public void Save(string fileName){
		using (var write = new StreamWriter(fileName)){

			write.WriteLine(_neuralNetworks.Length);
			write.WriteLine(_neuralNetworks[0].GetInputSize());
			write.WriteLine(_neuralNetworks[0].GetHiddenLayerSize());

			for (var i = 0; i < _neuralNetworks[0].GetHiddenLayerSize(); i++) write.WriteLine(_neuralNetworks[0].GetHiddenLayerSizeAt(i));

			write.WriteLine(_neuralNetworks[0].GetOutputSize());
		}
	}

	/// <summary>
	/// Loads Generational Network's parameters from a file
	/// </summary>
	/// <param name="fileName">Name of file to load from</param>
	public void Load(string fileName){
		using (var read = new StreamReader(fileName)){

			var neuroalNets = Convert.ToInt32(read.ReadLine());

			_neuralNetworks = new NeuralNetwork[neuroalNets];

			var inputSize = Convert.ToInt32(read.ReadLine());
			var hiddenLayerSize = Convert.ToInt32(read.ReadLine());
			var hidden = new int[Convert.ToInt32(read.ReadLine())];

			for(var i = 0;i < hiddenLayerSize;i++) hidden[i] = Convert.ToInt32(read.ReadLine());

			var outputSize = Convert.ToInt32(read.ReadLine());

			for(var i = 0;i < neuroalNets;i++) _neuralNetworks[i] = new NeuralNetwork(inputSize, hidden, outputSize);
		}
	}

	/// <summary>
	/// Rates neural network performence from 1..N
	/// By sorting from lowest error
	/// </summary>
	/// <param name="input">Input to neural networks</param>
	/// <param name="expected">Expected output</param>
	public void RateNeuralNetworks(double[] input, double[] expected){
		var errors = new double[_neuralNetworks.Length];

		for(var i = 0;i < errors.Length;i++) errors[i] = _neuralNetworks[i].Cost(input, expected);

		for(var i = 0;i < errors.Length;i++)
		for(var j = 0;j < errors.Length;j++)
			if(errors[i] < errors[j])
				_neuralNetworks[j].PerformanceNumber += 1;
	}


	/// <summary>
	/// Rates neural network performence from 1..N
	/// By sorting from lowest error
	/// </summary>
	/// <param name="input">Input to neural networks</param>
	/// <param name="expected">Expected output</param>
	public void RateNeuralNetworks(double[][] input, double[][] expected){
		var errors = new double[_neuralNetworks.Length];

		for(var i = 0;i < errors.Length;i++) errors[i] = _neuralNetworks[i].Cost(input, expected);

		for(var i = 0;i < errors.Length;i++)
		for(var j = 0;j < errors.Length;j++)
			if(errors[i] < errors[j])
				_neuralNetworks[j].PerformanceNumber += 1;
	}

	/// <summary>
	/// Returns best performance neural network
	/// </summary>
	public NeuralNetwork GetBestNetwork(){
		for(var i = 0;i < _neuralNetworks.Length;i++)
			if(_neuralNetworks[i].PerformanceNumber == 1) return _neuralNetworks[i];
		return null;
	}

	/// <summary>
	/// Saves the best Neural Network
	/// </summary>
	/// <param name="fileName">File to be saved to</param>
	public void SaveBestNetwork(string fileName) => GetBestNetwork().Save(fileName);

	/// <summary>
	/// Loads the best Neural Network
	/// </summary>
	/// <param name="fileName">File to be saved to</param>
	public void LoadBestNetwork(string fileName) => GetBestNetwork().Load(fileName);

	/// <summary>
	/// Return the average cost of all networks
	/// </summary>
	/// <param name="input">Input to networks to calculate error</param>
	/// <param name="expected">expected output to networks to calculate error</param>
	public double GetAverageCost(double[] input, double[] expected){
		double averageCost = 0;

		foreach (var t in _neuralNetworks) averageCost += t.Cost(input, expected);

		return averageCost / _neuralNetworks.Length;
	}

	/// <summary>
	/// Return the average cost of all networks
	/// </summary>
	/// <param name="input">Input to networks to calculate error</param>
	/// <param name="expected">expected output to networks to calculate error</param>
	public double GetAverageCost(double[][] input, double[][] expected){
		double averageCost = 0;

		foreach (var t in _neuralNetworks) averageCost += t.Cost(input, expected);

		return averageCost / _neuralNetworks.Length;
	}

	/// <summary>
	/// Return the lowest cost of all networks
	/// </summary>
	/// <param name="input">Input to networks to calculate error</param>
	/// <param name="expected">expected output to networks to calculate error</param>
	public double GetLowestCost(double[] input, double[] expected){
		var lowCost = _neuralNetworks[0].Cost(input, expected);

		for(var i = 1;i < _neuralNetworks.Length;i++)
			if(lowCost > _neuralNetworks[i].Cost(input, expected))
				lowCost = _neuralNetworks[i].Cost(input, expected);

		return lowCost;
	}

	/// <summary>
	/// Return the lowest cost of all networks
	/// </summary>
	/// <param name="input">Input to networks to calculate error</param>
	/// <param name="expected">expected output to networks to calculate error</param>
	public double GetLowestCost(double[][] input, double[][] expected){
		var lowCost = _neuralNetworks[0].Cost(input, expected);

		for(var i = 1;i < _neuralNetworks.Length;i++)
			if(lowCost > _neuralNetworks[i].Cost(input, expected))
				lowCost = _neuralNetworks[i].Cost(input, expected);

		return lowCost;
	}

	/// <summary>
	/// Selects best neural network and makes slightly changed copy's.
	/// Rate the networks before this
	/// </summary>
	public void NextGeneration(){
		var bestNetworkIndex = 0;

		for(var i = 0;i < _neuralNetworks.Length;i++)
			if(_neuralNetworks[i].PerformanceNumber == 1) bestNetworkIndex = i;

		for(var i = 0;i < _neuralNetworks.Length;i++){
			if(i == bestNetworkIndex) continue;
			_neuralNetworks[i].ApplyDataFromOtherNetwork(_neuralNetworks[bestNetworkIndex]);
			_neuralNetworks[i].SlightChange(_random);
		}
	}

	/// <summary>
	/// Selects best neural network and makes slightly changed copy's.
	/// Rate the networks before this
	/// </summary>
	/// <param name="multiplayer">The rate of change to copy networks</param>
	public void NextGeneration(double multiplayer){
		var bestNetworkIndex = 0;

		for(var i = 0;i < _neuralNetworks.Length;i++)
			if(_neuralNetworks[i].PerformanceNumber == 1) bestNetworkIndex = i;

		for(var i = 0;i < _neuralNetworks.Length;i++){
			if(i == bestNetworkIndex) continue;
			_neuralNetworks[i].ApplyDataFromOtherNetwork(_neuralNetworks[bestNetworkIndex]);
			_neuralNetworks[i].SlightChange(_random, multiplayer);
		}
	}

	public class NeuralNetwork
	{
		public  int         PerformanceNumber = 1;
		private double[]    _inputLayer;
		private double[][]  _hiddenLayer;
		private double[]    _output;
		private double[][]  _bias;
		private double[][]  _weights;

		public int GetInputSize()       => _inputLayer.Length;
		public int GetOutputSize()      => _output.Length;
		public int GetHiddenLayerSize() => _hiddenLayer.Length;
		public int GetHiddenLayerSizeAt(int index) => _hiddenLayer[index].Length;
		public double GetBias(int x, int y)        => _bias[x][y];
		public double GetWeight(int x, int y)      => _weights[x][y];

		public void ApplyDataFromOtherNetwork(NeuralNetwork network){
			for(var x = 0;x < _bias.Length;x++)
			for(var y = 0;y < _bias[x].Length;y++)
				_bias[x][y] = network.GetBias(x, y);

			for(var x = 0;x < _weights.Length;x++)
			for(var y = 0;y < _weights[x].Length;y++)
				_weights[x][y] = network.GetWeight(x, y);
		}

		public void SlightChange(Random rand)
		{
			foreach (var t in _bias)
				for(var y = 0;y < t.Length;y++)
					t[y] += (rand.NextDouble() - 0.5) * 0.01;

			foreach (var t in _weights)
				for(var y = 0;y < t.Length;y++)
					t[y] = (rand.NextDouble() - 0.5) * 0.01;
		}

		public void SlightChange(Random rand, double rate)
		{
			foreach (var t in _bias)
				for(var y = 0;y < t.Length;y++)
					t[y] += (rand.NextDouble() - 0.5) * rate;

			foreach (var t in _weights)
				for(var y = 0;y < t.Length;y++)
					t[y] = (rand.NextDouble() - 0.5) * rate;
		}

		public void Save(string fileName)
		{
			using (var write = new StreamWriter(fileName))
			{
				write.WriteLine(_inputLayer.Length);
				write.WriteLine(_output.Length);
				write.WriteLine(_hiddenLayer.Length);

				foreach (var t in _hiddenLayer) write.WriteLine(t.Length);

				write.WriteLine(_bias.Length);

				foreach (var t in _bias)
				{
					write.WriteLine(t.Length);
					foreach (var t1 in t) write.WriteLine(t1);
				}

				write.WriteLine(_weights.Length);

				foreach (var t in _weights)
				{
					write.WriteLine("{0}", t.Length);
					foreach (var t1 in t) write.WriteLine("{0}", t1);
				}
			}
		}

		public void Load(string fileName)
		{
			using (var read = new StreamReader(fileName))
			{
				_inputLayer = new double[Convert.ToInt32(read.ReadLine())];
				_output     = new double[Convert.ToInt32(read.ReadLine())];
				_hiddenLayer = new double[Convert.ToInt32(read.ReadLine())][];

				for (var i = 0; i < _hiddenLayer.Length; i++)
					_hiddenLayer[i] = new double[Convert.ToInt32(read.ReadLine())];

				_bias = new double[Convert.ToInt32(read.ReadLine())][];

				for (var i = 0; i < _bias.Length; i++)
				{
					_bias[i] = new double[Convert.ToInt32(read.ReadLine())];
					for (var j = 0; j < _bias[i].Length; j++) _bias[i][j] = Convert.ToDouble(read.ReadLine());
				}

				_weights = new double[Convert.ToInt32(read.ReadLine())][];

				for (var i = 0; i < _weights.Length; i++)
				{
					_weights[i] = new double[Convert.ToInt32(read.ReadLine())];
					for (var j = 0; j < _weights[i].Length; j++) _weights[i][j] = Convert.ToDouble(read.ReadLine());
				}
			}
		}

		public NeuralNetwork(int inputSize, int[] hiddenLayers, int outputSize)
		{
			_inputLayer     = new double[inputSize];
			_hiddenLayer    = new double[hiddenLayers.Length][];
			_bias           = new double[hiddenLayers.Length + 1][];

			for (var i = 0; i < hiddenLayers.Length; i++)
			{
				_hiddenLayer[i] = new double[hiddenLayers[i]];
				_bias[i] = new double[hiddenLayers[i]];
			}

			_bias[_bias.Length - 1] = new double[outputSize];
			_output         = new double[outputSize];
			_weights        = new double[2 + hiddenLayers.Length - 1][];
			_weights[0]     = new double[inputSize * hiddenLayers[0]];

			for (var i = 1; i < _weights.Length - 1; i++)
				_weights[i] = new double[hiddenLayers[i] * hiddenLayers[i - 1]];

			_weights[_weights.Length - 1] = new double[outputSize * hiddenLayers[hiddenLayers.Length - 1]];

			FirstRandomInit();
		}

		private void FirstRandomInit()
		{
			for (var y = 0; y < _weights.Length; y++)
			for (var x = 0; x < _weights[y].Length; x++)
				_weights[y][x] = (_random.NextDouble() - 0.5) * 2;

			for (var y = 0; y < _bias.Length; y++)
			for (var x = 0; x < _bias[y].Length; x++)
				_bias[y][x] = (_random.NextDouble() - 0.5) * 5;
		}

		public double[] GetOutput(double[] input)
		{
			_inputLayer = input;
			double sum;
			for (var i = 0; i < _hiddenLayer[0].Length; i++)
			{
				sum = 0;
				for (var j = 0; j < input.Length; j++) sum += _inputLayer[j] * _weights[0][j];

				sum += _bias[0][i];
				_hiddenLayer[0][i] = Sigmoid(sum);
			}

			for (var i = 1; i < _hiddenLayer.Length; i++)
			{
				sum = 0;
				for (var p = 0; p < _hiddenLayer[i].Length; p++)
				{
					for (var j = 0; j < _hiddenLayer[i - 1].Length; j++)
						sum += _hiddenLayer[i - 1][j] * _weights[i][j];

					sum += _bias[i][p];
					_hiddenLayer[i][p] = Sigmoid(sum);
				}
			}

			for (var i = 0; i < _output.Length; i++)
			{
				sum = 0;

				for (var j = 0; j < _hiddenLayer[_hiddenLayer.Length - 1].Length; j++)
					sum += _hiddenLayer[_hiddenLayer.Length - 1][j] * _weights[_weights.Length - 1][j];

				sum += _bias[_bias.Length - 1][i];
				_output[i] = Sigmoid(sum);
			}

			return _output;
		}

		private double Sigmoid(double value) => 1 / (1 + Math.Pow(Math.E, -value));

		public double Cost(double[] expected, double[] input)
		{
			double cost = 0;
			var value = GetOutput(input);

			for (var i = 0; i < expected.Length; i++) cost += Math.Pow(expected[i] - value[i], 2);

			return cost / expected.Length;
		}

		public double Cost(double[][] expected, double[][] input)
		{
			double cost = 0;
			var a = 0;

			for (var i = 0; i < expected.Length; i++)
			{
				var value = GetOutput(input[i]);

				for (var j = 0; j < expected[i].Length; j++)
				{
					cost += Math.Pow(expected[i][j] - value[j], 2);
					a++;
				}
			}

			return cost / a;
		}
	}
}
