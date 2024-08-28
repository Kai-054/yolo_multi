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