--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then
    require "initenv"
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-learn', false,
    'whether the network should learn or just make data')

cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', 'convnet_atari3', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', true,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 10^4, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 2*10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 50000000, 'number of training steps to perform')
cmd:option('-eval_steps', 250000, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')

cmd:text()
-- print(arg)
local opt = cmd:parse(arg)
-- print(opt)
-- local network_imported = (opt.network:sub(opt.network:len() - 2) == '.t7')

--- General setup.
game_env, game_actions, agent, opt = setup(opt)

-- print("agent: ".. tostring(agent))
-- print("agent network: ".. tostring(agent.network))
-- print(agent.network.modules[3].weight[1][1])
-- print("agent w: ".. tostring(agent.w))

-- for k, v in pairs(agent) do print(k) end



-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local total_reward
local nrewards
local nepisodes
local episode_reward



local batch_size = 30
local images = torch.Tensor(batch_size, 3, 210, 160)
local actions = torch.Tensor(batch_size)
local intra_batch_index = 1

local completed_train_batches = 0
local completed_test_batches = 0
local test_fraction = 10

local dataset_output_dir = 'dataset_' .. opt.name
os.execute('mkdir -p ' .. dataset_output_dir .. '/test')
os.execute('mkdir -p ' .. dataset_output_dir .. '/train')

if not opt.learn then ---------------------------------------------
    print("Generating data only. Not training.")
    local step = 0
    local screen, reward, terminal = game_env:getState()

    while step < opt.steps do
        step = step + 1

        local action_index = agent:perceive(reward, screen, terminal, true, 0.1)

        local save_flag = true
        if intra_batch_index == 1 then
            if torch.random(10) ~= 1 then
                save_flag = false
            end
        end

        if save_flag then
            images[intra_batch_index] = screen:float()
            actions[intra_batch_index] = action_index
            intra_batch_index = intra_batch_index + 1

            if intra_batch_index > batch_size then
                if torch.random(test_fraction) == 1 then
                    completed_test_batches = completed_test_batches + 1
                    torch.save(dataset_output_dir .. '/test/images_batch_' .. completed_test_batches, images)
                    torch.save(dataset_output_dir .. '/test/actions_batch_' .. completed_test_batches, actions)
                else
                    completed_train_batches = completed_train_batches + 1
                    torch.save(dataset_output_dir .. '/train/images_batch_' .. completed_train_batches, images)
                    torch.save(dataset_output_dir .. '/train/actions_batch_' .. completed_train_batches, actions)
                end

                intra_batch_index = 1
            end
        end

        if not terminal then
            screen, reward, terminal = game_env:step(game_actions[action_index], true)
        else
            if opt.random_starts > 0 then
                screen, reward, terminal = game_env:nextRandomGame()
            else
                screen, reward, terminal = game_env:newGame()
            end
        end

        if step%1000 == 0 then collectgarbage() end
    end



else ----------------------------------------------

    --

    -- if network_imported then
    --     local loaded_w = torch.load(opt.network:sub(1, opt.network:len() - 3) .. '.params.t7', 'ascii').network
    --     agent.w = loaded_w:cuda()
    -- end

    local screen, reward, terminal = game_env:getState()

    print("Iteration ..", step)
    while step < opt.steps do
        step = step + 1
        local action_index = agent:perceive(reward, screen, terminal)

        -- only save every now and then,
        -- but make sure the batches stay whole! (sequential)
        local save_flag = false
        if intra_batch_index == 1 then
            if torch.random(1000) ~= 1 then
                save_flag = false
            end
        end

        if save_flag then
            images[intra_batch_index] = screen:float()
            actions[intra_batch_index] = action_index
            intra_batch_index = intra_batch_index + 1

            if intra_batch_index > batch_size then
                if torch.random(test_fraction) == 1 then
                    completed_test_batches = completed_test_batches + 1
                    torch.save(dataset_output_dir .. '/test/images_batch_' .. completed_test_batches, images)
                    torch.save(dataset_output_dir .. '/test/actions_batch_' .. completed_test_batches, actions)
                else
                    completed_train_batches = completed_train_batches + 1
                    torch.save(dataset_output_dir .. '/train/images_batch_' .. completed_train_batches, images)
                    torch.save(dataset_output_dir .. '/train/actions_batch_' .. completed_train_batches, actions)
                end

                intra_batch_index = 1
                -- images = torch.Tensor(batch_size, 3, 210, 160)
                -- actions = torch.Tensor(batch_size)
            end
        end


        -- game over? get next game!
        if not terminal then
            screen, reward, terminal = game_env:step(game_actions[action_index], true)
        else
            if opt.random_starts > 0 then
                screen, reward, terminal = game_env:nextRandomGame()
            else
                screen, reward, terminal = game_env:newGame()
            end
        end

        if step % opt.prog_freq == 0 then
            -- print(screen)
            assert(step==agent.numSteps, 'trainer step: ' .. step ..
                    ' & agent.numSteps: ' .. agent.numSteps)
            print("Steps: ", step)
            agent:report()
            collectgarbage()
        end

        if step%1000 == 0 then collectgarbage() end

        if false then -- step % opt.eval_freq == 0 and step > learn_start then

            screen, reward, terminal = game_env:newGame()

            total_reward = 0
            nrewards = 0
            nepisodes = 0
            episode_reward = 0

            local eval_time = sys.clock()
            for estep=1,opt.eval_steps do
                local action_index = agent:perceive(reward, screen, terminal, true, 0.05)

                -- Play game in test mode (episodes don't end when losing a life)
                screen, reward, terminal = game_env:step(game_actions[action_index])

                if estep%1000 == 0 then collectgarbage() end

                -- record every reward
                episode_reward = episode_reward + reward
                if reward ~= 0 then
                   nrewards = nrewards + 1
                end

                if terminal then
                    total_reward = total_reward + episode_reward
                    episode_reward = 0
                    nepisodes = nepisodes + 1
                    screen, reward, terminal = game_env:nextRandomGame()
                end
            end

            eval_time = sys.clock() - eval_time
            start_time = start_time + eval_time
            agent:compute_validation_statistics()
            local ind = #reward_history+1
            total_reward = total_reward/math.max(1, nepisodes)

            if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
                agent.best_network = agent.network:clone()
            end

            if agent.v_avg then
                v_history[ind] = agent.v_avg
                td_history[ind] = agent.tderr_avg
                qmax_history[ind] = agent.q_max
            end
            print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])

            reward_history[ind] = total_reward
            reward_counts[ind] = nrewards
            episode_counts[ind] = nepisodes

            time_history[ind+1] = sys.clock() - start_time

            local time_dif = time_history[ind+1] - time_history[ind]

            local training_rate = opt.actrep*opt.eval_freq/time_dif

            print(string.format(
                '\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' ..
                'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
                'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d',
                step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif,
                training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
                nepisodes, nrewards))
        end

        if step % opt.save_freq == 0 or step == opt.steps then
           print(agent.network.modules[3].weight[1][1])
            local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
                agent.valid_s2, agent.valid_term
            agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
                agent.valid_term = nil, nil, nil, nil, nil, nil, nil
            local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
                agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
            agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
                agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

            local filename = opt.name
            if opt.save_versions > 0 then
                filename = filename .. "_" .. math.floor(step / opt.save_versions)
            end
            os.execute('mkdir -p networks')
            filename = paths.concat('networks', filename)
            torch.save(filename .. ".t7", {agent = agent,
                                    model = agent.network,
                                    best_model = agent.best_network,
                                    reward_history = reward_history,
                                    reward_counts = reward_counts,
                                    episode_counts = episode_counts,
                                    time_history = time_history,
                                    v_history = v_history,
                                    td_history = td_history,
                                    qmax_history = qmax_history,
                                    arguments=opt})
            if opt.saveNetworkParams then
                local nets = {network=w:clone():float()}
                torch.save(filename..'.params.t7', nets, 'ascii')
                print('Saved:', filename .. '.params.t7')
            end
            agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
                agent.valid_term = s, a, r, s2, term
            agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
                agent.deltas, agent.tmp = w, dw, g, g2, delta, delta2, deltas, tmp
            print('Saved:', filename .. '.t7')
            io.flush()
            collectgarbage()
        end
    end
end
