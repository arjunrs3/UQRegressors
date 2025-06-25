    def _fit_single_model(self, X_tensor, y_tensor, X_cal, y_cal, mode="FIT"): 
        if mode == "FIT": 
            epochs = self.epochs 
        elif mode == "BO": 
            epochs = self.tune_epochs

        config = {
            "learning_rate": self.learning_rate,
            "epochs": epochs,
            "batch_size": self.batch_size,
        }

        logger = Logger(
            use_wandb=self.use_wandb,
            project_name=self.wandb_project,
            run_name=self.wandb_run_name,
            config=config,
        )

        activation = get_activation(self.activation_str)

        self.model = MLP(self.input_dim, self.hidden_sizes, self.dropout, activation)
        self.model.to(self.device)

        optimizer = self.optimizer_cls(
            self.model.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
        )

        scheduler = None
        if self.scheduler_cls is not None:
            scheduler = self.scheduler_cls(optimizer, **self.scheduler_kwargs)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in dataloader: 
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss

            if scheduler is not None:
                scheduler.step()

            if epoch % (epochs / 20) == 0:
                logger.log({"epoch": epoch, "train_loss": epoch_loss})

        oof_preds = self.model(X_cal)
        loss_matrix = (oof_preds - y_cal) * torch.tensor([1, -1], device=self.device)
        print(oof_preds[-10:])
        print(y_cal[-10:])
        print(loss_matrix[-10:])
        self.residuals = torch.max(loss_matrix, dim=1).values
        n = len(self.residuals)
        q = int((1 - self.alpha) * (n + 1))
        q = min(max(q, 0), n-1)
        self.conformal_width = torch.topk(self.residuals, n-q).values[-1].detach().cpu().numpy()

        logger.finish()

    def fit(self, X, y): 
        if self.random_seed is not None: 
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

        if self.scale_data: 
            X = self.input_scaler.fit_transform(X)
            y = self.output_scaler.fit_transform(y.reshape(-1, 1))

        X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=self.cal_size, random_state=self.random_seed)
        
        if self.tune_quantiles or self.tune_epochs: 
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.validation_size, random_state=self.random_seed)


        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        input_dim = X.shape[1]
        self.input_dim = input_dim 

        X_cal = torch.tensor(X_cal, dtype=torch.float32).to(self.device)
        y_cal = torch.tensor(y_cal, dtype=torch.float32).to(self.device)

        if self.tune_quantiles or self.tune_epochs: 
            best_hyperparams = self._tune_hyperparams(self, X_tensor, y_tensor, X_cal, y_cal, X_val, y_val)
            self.tau_lo = best_hyperparams[0] or self.tau_lo 
            self.tau_hi = best_hyperparams[1] or self.tau_hi 
            self.epochs = best_hyperparams[2] or self.epochs

        return self

    def _tune_hyperparams(self, X_train, y_train, X_cal, y_cal, X_val, y_val):
        