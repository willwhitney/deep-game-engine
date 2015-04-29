-- performs a finite differences test on the forward and backward
-- methods of a module to ensure that the gradients returned by
-- backward match the "true" gradients of the forward function.

function finiteDiff(mod, input, target)
    epsilon = 1e-4
    fdgrad = {torch.zeros(5, 1):cuda(), torch.zeros(5, 1):cuda()}

    for batch = 1, input[1]:size(1) do
        for i = 1, input[1]:size(2) do
            mu = input[1]:clone()
            mu[batch][i] = mu[batch][i] - epsilon
            f_minus = mod:forward({mu, input[2]}, target)

            mu = input[1]:clone()
            mu[batch][i] = mu[batch][i] + epsilon
            f_plus = mod:forward({mu, input[2]}, target)

            fdgrad[1][batch][i] = (f_plus - f_minus) / (2 * epsilon)
        end
    end

    for batch = 1, input[2]:size(1) do
        for i = 1, input[2]:size(2) do
            sigma = input[2]:clone()
            sigma[batch][i] = sigma[batch][i] - epsilon
            f_minus = mod:forward({input[1], sigma}, target)

            sigma = input[2]:clone()
            sigma[batch][i] = sigma[batch][i] + epsilon
            f_plus = mod:forward({input[1], sigma}, target)

            fdgrad[2][batch][i] = (f_plus - f_minus) / (2 * epsilon)
        end
    end

    mod:forward(input, target)
    modulegrad = mod:backward(input, target)


    print("module mu gradient ↓")
    print(simplestr(modulegrad[1]))
    print(simplestr(fdgrad[1]))
    print("TRUE mu gradient ↑")

    print("module sigma gradient ↓")
    print(simplestr(modulegrad[2]))
    print(simplestr(fdgrad[2]))
    print("TRUE sigma gradient ↑")

    return fdgrad
end

