function C = digit_classify(mdl, testdata)
%Function C = digit_classify(mdl, testdata) takes input as a matrix N*3 data sample
% of an air-written digit collected by LeapMotion sensor and does the 
% recognition of the written digit.
%Input:
%   testdata: a matrix N*3 data sample (N number of 3-D location datapoint trajectories)
%Output:
%   C: The label of the written digit predicted by the provided model
    
    %% Turn sample data from a matrix to a cell array with one cell because
    % of input format of called functions
    test_datacell ={};
    test_datacell{1,1} = testdata;
    % number of timesteps (data points) of the given to-be-classified sample
    nTimesteps = size(test_datacell{1},1); 
    % Load data.mat
    load data.mat data class
    %% Step 1: Normalize data
    test_datacell = data_normalization(test_datacell);
    
    %% Step 2: Extract features to ensure consistent size of data sample and
    model_data_size = size(data,2)/3;
    % If the input testdata sample has more timesteps than the
    % data samples used to train the model, the testdata sample needs to
    % proceed feature extraction
    if nTimesteps > model_data_size
        test_datacell = feature_extraction(test_datacell, model_data_size);
    % SPECIAL CASE: MODEL RETRAINING NEEDED
    elseif nTimesteps < model_data_size %UPSAMPLING???
        % There is no need to extract feature for the given test data
        % sample because its number of datapoints is the new standard
%         test_datacell = feature_extraction(test_datacell, nTimesteps);
        % Repreprocess raw data
        load raw_data.mat raw_data class
        normalized_traindata = data_normalization(raw_data);
        extracted_traindata = feature_extraction(normalized_traindata,nTimesteps);
        train_data = data_reallocation(extracted_traindata);
        [coeff, trainX, score, explained, mu, toKeepComponentsIdx] = pca_implementation(train_data, 98);
        % Retrain model and identify parameters
        mdl_method = class(mdl);
        if mdl_method == 'ClassificationECOC'
            disp('SVM Classification model retraining...')
%TODO
        end
        % check accuracy on traindata test set and print prompt
%         fprintf("The accuracy of the model is %d.\n", sum(net_testclass==testclass_)/length(testclass_)); 
    end
    %% Step 3: Reallocate test data sample (turn a cell array with one cell into a column vector)
    test_datavec = data_reallocation(test_datacell);
    load pca_output.mat
    testX = pca_tranformation(testdata, coeff, mu, toKeepComponentsIdx);
    %% Step 4: Predict label for input test data sample
    if nTimesteps < model_data_size
        % SPECIAL CASE: USE RETRAINED PARAMETERS
        C = predict(new_mdl,test_datavec);
    else
        % NORMAL CASE: USE TRAINED PARAMETERS
        C = predict(mdl,test_datavec);
    end
    
       
    