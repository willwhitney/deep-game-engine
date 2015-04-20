-- require 'nn'
local PrintModule, parent = torch.class('nn.PrintModule', 'nn.Module')

function PrintModule:__init(name)
  self.name = name
end

function PrintModule:updateOutput(input)
  print(self.name.." input dimensions: ")
  if type(input) == 'table' then
    print("table:", input)
  else
    print(input:size())
  end
  self.output = input
  return input
end

function PrintModule:updateGradInput(input, gradOutput)
  print(self.name.." gradInput dimensions:")
  if type(gradOutput) == 'table' then
    print("table:", #gradOutput)
  else
    print(gradOutput:size())
  end
  self.gradInput = gradOutput
  return self.gradInput
end

