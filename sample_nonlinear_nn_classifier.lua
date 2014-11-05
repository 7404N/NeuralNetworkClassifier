require('torch')
require('nn')
require('math')

dimensions = 10 --input vector size

function testModel(input)
  	local result =  model:forward(input)
  	print( 'multiply is '.. input[1]*input[2])
  	print(result:apply(math.exp))
end

function genTrainset()
	dataset = {}
	mean = {}
	std = {}
	
	function dataset:size()
		return 1000
	end

	inputs = torch.randn(dataset:size(),dimensions)
	outputs = {}

	for i=1, dataset:size() do
		local input = inputs[i]
		local output = 1
		local multi = 1

		--replace with your code here
		for j = 1, 2 do multi = multi*input[j] end;
		
		if multi < 0.2 then
			output = 1
		elseif multi < 0.4 then
			output = 2
		elseif multi < 0.6 then
			output = 3
		elseif multi < 0.8 then
			output = 4
		else
			output = 5
		end

	outputs[i] = output;
	end

	--normalize
	--for j=1, dimensions do
	--	local m = inputs[{{},{j}}]:mean()
	--	local div = inputs[{{},{j}}]:std()
	--	inputs[{{},{j}}]:add(-m)
	--	inputs[{{},{j}}]:div(div)
	--end

	for i=1,dataset:size() do
		dataset[i] = {inputs[i], outputs[i]}
	end
end

dataset = {}

function trainMClassifier()
	genTrainset()

	hiddenCount = 256
	classes = 5 
	model = nn.Sequential();
	model:add(nn.Linear(10, hiddenCount))
	model:add(nn.Tanh())
	model:add(nn.Linear(hiddenCount,classes))
	model:add(nn.LogSoftMax())

	criterion = nn.ClassNLLCriterion()  
	trainer = nn.StochasticGradient(model, criterion)
	trainer.learningRate = 0.01
	trainer.maxIteration = 100
	trainer:train(dataset)
end

--train
trainMClassifier()

--test
for i=1,100 do
	testModel(torch.randn(dimensions))
end
