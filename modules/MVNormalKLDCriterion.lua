--[[
Computes the KLD of the input distribution from the target distribution.

Assumes that the input and target distributions are multivariate Gaussian
with diagonal covariance matrices.

Input and target should each be specified as {mu, sigma} where mu and sigma
are tensors.
--]]

local MVNormalKLDCriterion, parent = torch.class('nn.MVNormalKLDCriterion', 'nn.Criterion')

function MVNormalKLDCriterion:__init()
   parent.__init(self)
end

function MVNormalKLDCriterion:updateOutput(input, target)
    self.dimension = input[1]:size(2)

    -- since in KLDCriterion we're actually learning log(sigma^2)
    -- since we actually need sigma, we need to exp and sqrt
    self.mu_q       = input[1]
    self.sigma_q    = self.sigma_q or torch.Tensor(input[2]:size()):cuda()
    self.sigma_q    = self.sigma_q:copy(input[2]):exp():pow(1/2)

    self.mu_p      = target[1]
    self.sigma_p   = self.sigma_p or torch.Tensor(target[2]:size()):cuda()
    self.sigma_p   = self.sigma_p:copy(target[2]):exp():pow(1/2)

    -- log |\Sigma_q| / |\Sigma_p|
    -- this should be equivalent and doesn't have underflow issues
    self.term1 = torch.sum(torch.log(self.sigma_q), 2) - torch.sum(torch.log(self.sigma_p), 2)
    -- self.term1 = torch.log(torch.prod(self.sigma_q, 2)) - torch.log(torch.prod(self.sigma_p, 2))
    -- print("term1:", self.term1:norm())

    -- tr(\Sigma_q^{-1} * \Sigma_p)
    self.term2 = torch.sum(torch.cdiv(self.sigma_p, self.sigma_q), 2)
    -- print("term2:", self.term2:norm())

    -- (\mu_q - \mu_p)^T * \Sigma_q^{-1} * (\mu_q - \mu_p)
    self.term3 = torch.sum(torch.cdiv(torch.pow(self.mu_q - self.mu_p, 2), self.sigma_q), 2)
    -- print("term3:", self.term3:norm())

    self.output = torch.sum((self.term1 - self.dimension + self.term2 + self.term3)) * 1/2
    return self.output
end

function MVNormalKLDCriterion:updateGradInput(target, input)
    self.mu_q       = input[1]
    self.sigma_q    = self.sigma_q or torch.Tensor(input[2]:size()):cuda()
    self.sigma_q    = self.sigma_q:copy(input[2]):exp():pow(1/2)

    self.mu_p      = target[1]
    self.sigma_p   = self.sigma_p or torch.Tensor(target[2]:size()):cuda()
    self.sigma_p   = self.sigma_p:copy(target[2]):exp():pow(1/2)

    self.grad_mu = torch.cdiv(self.mu_p - self.mu_q, self.sigma_q)
    self.grad_sigma = (torch.pow(self.sigma_q, -1) - torch.pow(self.sigma_p, -1)) * 1/2

    -- deal with the fact that we're actually learning log(sigma^2)
    self.d_sigma_d_input_2 = torch.cmul(torch.pow(torch.exp(input[2]), -1/2), torch.exp(input[2])) * 1/2
    self.grad_input_2 = torch.cmul(self.d_sigma_d_input_2, self.grad_sigma)

    self.gradInput = {
            self.grad_mu,
            self.grad_input_2
        }

    return self.gradInput
end
