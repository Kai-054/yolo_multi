- task.py
 """
  outputs, y = [], []
        print ("outputs=", outputs)
        print("Y=================", y)
        exit()

        for m in self.model:  # m đại diện cho 1 layer
            if m.f != -1:  # nếu không phải từ lớp trước
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                print("x_1****************", x)
            x = m(x)  # run
            print("x_2*********************", x)
 """
        Ý nghĩa của x trong vòng lặp for
        Ban đầu:

        x là tensor đầu vào (input image tensor), tức là dữ liệu hình ảnh đã được chuyển đổi thành tensor trước khi đưa vào mô hình.
        Trong vòng lặp for:

        Vòng lặp for chạy qua tất cả các layer trong mô hình (self.model).
        m đại diện cho một layer cụ thể trong mô hình.
        x được truyền qua mỗi layer m để tạo ra đầu ra của layer đó.
        Cập nhật x qua mỗi layer:

        Ở mỗi bước trong vòng lặp, x được cập nhật với đầu ra từ layer hiện tại.
        x_1: Trước khi x đi qua layer hiện tại, x là đầu ra từ các layer trước đó hoặc có thể là tổ hợp của nhiều đầu ra từ các layer khác nhau (nếu m.f là một danh sách).
        x_2: Sau khi x đi qua layer m, nó trở thành đầu ra của layer đó và tiếp tục được truyền vào layer tiếp theo trong vòng lặp.
        Tính linh hoạt của x:

        x có thể là đầu ra từ một hoặc nhiều layer trước đó. Nếu m.f != -1, x được lấy từ các layer được chỉ định bởi m.f. Điều này có nghĩa là x không nhất thiết phải là đầu ra từ layer ngay trước đó, mà có thể là từ các layer khác trước đó trong mô hình.
        x sau đó được đưa vào layer m để thực hiện các phép biến đổi, và giá trị này được sử dụng làm đầu vào cho layer tiếp theo trong vòng lặp.


1. Thiết lập ban đầu
python
Sao chép mã
if world_size > 1:
    self._setup_ddp(world_size)

self._setup_train(world_size)
Nếu quá trình huấn luyện diễn ra trong môi trường phân tán (ví dụ, có nhiều hơn 1 GPU, world_size > 1), hàm _setup_ddp sẽ khởi tạo môi trường huấn luyện song song phân tán.
Hàm _setup_train thiết lập các cấu hình cần thiết cho quá trình huấn luyện dựa trên số thiết bị đang sử dụng (có thể là GPU hoặc CPU).
2. Thiết lập thời gian và tham số huấn luyện
python
Sao chép mã
self.epoch_time = None
self.epoch_time_start = time.time()
self.train_time_start = time.time()
nb = len(self.train_loader)  # số lượng batch
nw = max(round(self.args.warmup_epochs * nb), 100)  # số lượng warmup iterations
last_opt_step = -1
epoch_time_start và train_time_start lưu lại thời gian bắt đầu của epoch và quá trình huấn luyện để đo thời gian.
nb là số lượng batch trong một epoch (một epoch là một lần duyệt toàn bộ dữ liệu).
nw là số lượng lần lặp (iterations) cho giai đoạn "warmup" (khi huấn luyện mô hình dần dần tăng tốc độ học).
3. Bắt đầu huấn luyện
python
Sao chép mã
for epoch in range(self.start_epoch, self.epochs):
    self.epoch = epoch
    self.run_callbacks('on_train_epoch_start')
    self.model.train()  # Đặt mô hình trong trạng thái huấn luyện
    if RANK != -1:
        self.train_loader.sampler.set_epoch(epoch)
Vòng lặp for chạy qua các epoch từ self.start_epoch đến self.epochs (số epoch cần huấn luyện).
self.run_callbacks('on_train_epoch_start') gọi các hàm callback (hàm thực thi trước khi bắt đầu epoch).
self.model.train() đặt mô hình vào chế độ huấn luyện, điều này rất quan trọng vì nó kích hoạt các cơ chế như dropout.
Nếu RANK != -1, có nghĩa là mô hình đang huấn luyện phân tán, và set_epoch đảm bảo dữ liệu được lấy mẫu (sampling) ngẫu nhiên cho mỗi epoch.
4. Tạo thanh tiến trình và đóng mosaic
python
Sao chép mã
if epoch == (self.epochs - self.args.close_mosaic):
    LOGGER.info('Closing dataloader mosaic')
    if hasattr(self.train_loader.dataset, 'mosaic'):
        self.train_loader.dataset.mosaic = False
    if hasattr(self.train_loader.dataset, 'close_mosaic'):
        self.train_loader.dataset.close_mosaic(hyp=self.args)
    self.train_loader.reset()
Khi đến một epoch nhất định (xác định bởi close_mosaic), hàm này tắt chế độ mosaic (một kỹ thuật để kết hợp nhiều hình ảnh nhỏ lại với nhau, giúp mô hình có thêm nhiều biến thể dữ liệu).
self.train_loader.reset() sẽ reset lại dataloader sau khi tắt mosaic.
5. Vòng lặp qua các batch trong một epoch
python
Sao chép mã
for i, batch in pbar:
    self.run_callbacks('on_train_batch_start')
Vòng lặp for qua từng batch dữ liệu trong dataloader (self.train_loader).
Mỗi batch được lấy ra và callback on_train_batch_start được gọi trước khi bắt đầu huấn luyện batch đó.
6. Warmup (khởi động)
python
Sao chép mã
if ni <= nw:
    xi = [0, nw]
    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
    for j, x in enumerate(self.optimizer.param_groups):
        x['lr'] = np.interp(ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
        if 'momentum' in x:
            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
Giai đoạn "warmup" điều chỉnh các tham số như learning rate (tốc độ học) và momentum dần dần để tránh việc mô hình học quá nhanh và gây ra mất ổn định.
np.interp sử dụng để nội suy các giá trị của tốc độ học và momentum từ một giá trị thấp đến giá trị mong muốn trong quá trình warmup.
7. Tiến hành dự đoán và tính toán loss
python
Sao chép mã
with torch.cuda.amp.autocast(self.amp):
    batch = self.preprocess_batch(batch)
    if self.args.task == "multi":
        preds = self.model(batch[0]['img'])
        # Tính toán loss cho mỗi batch
        for count in range(len(batch)):
            self.mul_loss[count], self.mul_loss_items[count] = self.criterion(preds[count], batch[count], self.data['labels_list'][count], count)
        self.loss = sum(self.mul_loss)  # Tổng hợp loss từ nhiều batch
    else:
        preds = self.model(batch['img'])
        self.loss, self.loss_items = self.criterion(preds, batch)
Nếu mô hình hỗ trợ nhiều tác vụ (task), các batch được xử lý qua model và tính toán loss riêng cho từng batch.
Loss của từng batch sẽ được tổng hợp thành self.loss.
Nếu không có nhiều tác vụ, chỉ một tác vụ sẽ được xử lý như thông thường.
8. Lan truyền ngược và tối ưu hóa
python
Sao chép mã
self.scaler.scale(self.loss).backward(retain_graph=False)
self.optimizer_step()
self.loss được sử dụng để tính toán gradient thông qua hàm backward(), sau đó hàm optimizer_step() sẽ thực hiện cập nhật các tham số mô hình dựa trên gradient tính toán được.
retain_graph=False giúp giải phóng bộ nhớ GPU sau khi gradient đã được tính toán xong.
9. Đăng nhập và hiển thị tiến trình
python
Sao chép mã
if RANK in (-1, 0):
    pbar.set_description(...)
    self.run_callbacks('on_batch_end')
Thanh tiến trình (pbar) hiển thị các thông tin như loss, số lượng lớp, và kích thước ảnh trong batch hiện tại.
Callback on_batch_end được gọi sau mỗi batch để thực hiện các tác vụ bổ sung như ghi log hoặc kiểm tra điều kiện dừng.
10. Kiểm tra và lưu kết quả sau mỗi epoch
python
Sao chép mã
if self.args.val or final_epoch:
    self.metrics, self.fitness = self.validate()
self.save_model()
Sau mỗi epoch, nếu cờ val được bật hoặc đây là epoch cuối, mô hình sẽ được đánh giá trên tập dữ liệu kiểm tra (validate()).
Nếu epoch đó là epoch cuối hoặc có yêu cầu, mô hình sẽ được lưu (save_model()).
11. Dừng sớm và dọn dẹp
python
Sao chép mã
if self.stop:
    break  # Phải dừng tất cả các node trong DDP
Nếu điều kiện dừng sớm được kích hoạt (self.stop), quá trình huấn luyện sẽ dừng lại.
Sau khi huấn luyện xong, GPU sẽ được giải phóng bộ nhớ (torch.cuda.empty_cache()).