
-- this is exactly the torch/nn Replicate
-- copied here because it has had breaking changes
local ReplicateLocal, parent = torch.class('nn.ReplicateLocal','nn.Module')

function ReplicateLocal:__init(nf, dim)
   parent.__init(self)
   self.nfeatures = nf
   self.dim = dim or 1
   assert(self.dim > 0, "Can only replicate across positive integer dimensions.")
end

function ReplicateLocal:updateOutput(input)
   self.dim = self.dim or 1 --backwards compatible
   assert(
      self.dim <= input:dim()+1,
      "Not enough input dimensions to replicate along dimension " ..
      tostring(self.dim) .. ".")
   local sz = torch.LongStorage(input:dim()+1)
   sz[self.dim] = self.nfeatures
   for i = 1,input:dim() do
      local offset = 0
      if i >= self.dim then
         offset = 1
      end
      sz[i+offset] = input:size(i)
   end
   local st = torch.LongStorage(input:dim()+1)
   st[self.dim] = 0
   for i = 1,input:dim() do
      local offset = 0
      if i >= self.dim then
         offset = 1
      end
      st[i+offset] = input:stride(i)
   end
   self.output = input.new(input:storage(),input:storageOffset(),sz,st)
   return self.output
end

function ReplicateLocal:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   local sz = torch.LongStorage(input:dim()+1)
   sz[self.dim] = 1
   for i = 1,input:dim() do
      local offset = 0
      if i >= self.dim then
         offset = 1
      end
      sz[i+offset] = input:size(i)
   end
   local gradInput = self.gradInput:view(sz)
   gradInput:sum(gradOutput, self.dim)
   return self.gradInput
end