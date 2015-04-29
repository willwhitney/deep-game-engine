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

    -- dpv = pv.prod()
    -- dqv = qv.prod(axis)
    -- # Inverse of diagonal covariance qv
    -- iqv = 1./qv
    -- # Difference between means pm, qm
    -- diff = qm - pm
    -- return (0.5 *
            -- (numpy.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
            --  + (iqv * pv).sum(axis)          # + tr(\Sigma_q^{-1} * \Sigma_p)
            --  + (diff * iqv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
            --  - len(pm)))                     # - N

-- I think reversing these did the trick?

function MVNormalKLDCriterion:updateOutput(target, input)
    self.dimension = input[1]:size(2)

    -- as per KLDCriterion, we're actually learning log(sigma^2)
    -- since we actually need sigma, we need to exp and sqrt
    self.mu_input       = input[1]
    self.sigma_input    = self.sigma_input or torch.Tensor(input[2]:size()):cuda()
    self.sigma_input    = self.sigma_input:copy(input[2]):exp():pow(1/2)

    self.mu_target      = target[1]
    self.sigma_target   = self.sigma_target or torch.Tensor(target[2]:size()):cuda()
    self.sigma_target   = self.sigma_target:copy(target[2]):exp():pow(1/2)

    -- print("sigma input:", self.sigma_input:norm())
    -- print("sigma target:", self.sigma_target:norm())

    -- log |\Sigma_q| / |\Sigma_p|
    self.term1 = torch.log(torch.prod(self.sigma_target, 2)) - torch.log(torch.prod(self.sigma_input, 2))
    -- print("term1:", self.term1:norm())

    -- tr(\Sigma_q^{-1} * \Sigma_p)
    self.term2 = torch.sum(torch.cdiv(self.sigma_input, self.sigma_target), 2)
    -- print("term2:", self.term2:norm())

    -- (\mu_q - \mu_p)^T * \Sigma_q^{-1} * (\mu_q - \mu_p)
    self.term3 = torch.sum(torch.cdiv(torch.pow(self.mu_target - self.mu_input, 2), self.sigma_target), 2)
    -- print("term3:", self.term3:norm())

    self.output = torch.sum((self.term1 - self.dimension + self.term2 + self.term3)) * 1/2
    return self.output
end

function MVNormalKLDCriterion:updateGradInput(target, input)
    self.mu_input       = input[1]
    self.sigma_input    = self.sigma_input or torch.Tensor(input[2]:size()):cuda()
    self.sigma_input    = self.sigma_input:copy(input[2]):exp():pow(1/2)

    self.mu_target      = target[1]
    self.sigma_target   = self.sigma_target or torch.Tensor(target[2]:size()):cuda()
    self.sigma_target   = self.sigma_target:copy(target[2]):exp():pow(1/2)

    self.grad_mu = torch.cdiv(self.mu_input - self.mu_target, self.sigma_target)
    self.grad_sigma = (torch.pow(self.sigma_target, -1) - torch.pow(self.sigma_input, -1)) * 1/2

    -- deal with the fact that we're actually learning log(sigma^2)
    self.d_sigma_d_input_2 = torch.cmul(torch.pow(torch.exp(input[2]), -1/2), torch.exp(input[2])) * 1/2
    self.grad_input_2 = torch.cmul(self.d_sigma_d_input_2, self.grad_sigma)

    self.gradInput = {
            self.grad_mu,
            self.grad_input_2
        }

    return self.gradInput
end
