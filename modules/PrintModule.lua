-- require 'nn'
local PrintModule, parent = torch.class('nn.PrintModule', 'nn.Module')

function PrintModule:__init(name)
  self.name = name
end

function PrintModule:updateOutput(input)
  print(self.name.." input dimensions: ")
  print(input:size())
  self.output = input
  return input
end

function PrintModule:updateGradInput(input, gradOutput)
  print(self.name.." gradInput dimensions:")
  print(gradOutput:size())
  self.gradInput = gradOutput
  return self.gradInput
end

