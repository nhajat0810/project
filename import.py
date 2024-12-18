'''import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

# Đường dẫn tới thư mục chứa dữ liệu huấn luyện và kiểm tra
train_dir = "C:/Users/LapTop/Downloads/pro3/train"
test_dir = "C:/Users/LapTop/Downloads/pro3/test"
img_size = 48

# Cài đặt ImageDataGenerator cho dữ liệu huấn luyện và kiểm tra
train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rotation_range=20,  
    zoom_range=0.3,     
    shear_range=0.3,    
    brightness_range=[0.8, 1.2],  # Thêm biến đổi độ sáng
    channel_shift_range=50.0,     # Thêm biến đổi kênh màu
    rescale=1./255,
    validation_split=0.2  
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Tạo generator cho dữ liệu huấn luyện và kiểm tra
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(img_size, img_size),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training"
)

validation_generator = validation_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(img_size, img_size),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)

# Learning Rate Scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_size, img_size, 1), kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(128, (5, 5), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.4),

    Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.4),

    GlobalAveragePooling2D(),  # Thay Flatten bằng GAP để giảm overfitting
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(7, activation='softmax')
])

# Biên dịch mô hình
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback
checkpoint = ModelCheckpoint(
    'model_best.keras', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max'
)

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=8,
    restore_best_weights=True
)

lr_callback = LearningRateScheduler(lr_scheduler)

# Huấn luyện mô hình
epochs = 100
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping, lr_callback]
)

# Vẽ biểu đồ độ chính xác và độ mất mát
fig, ax = plt.subplots(1, 2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
fig.set_size_inches(12, 4)

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Training Accuracy vs Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Training Loss vs Validation Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Lưu mô hình cuối cùng
model.save('model_final_improved.h5')
'''
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Đường dẫn tới thư mục chứa dữ liệu huấn luyện và kiểm tra
train_dir = "../pro3/train"
test_dir = "../pro3/test"
img_size = 48

# Cài đặt ImageDataGenerator cho dữ liệu huấn luyện và kiểm tra
train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rescale=1./255,
    validation_split=0.2  # Sử dụng 20% dữ liệu cho validation
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Dữ liệu cho validation sẽ lấy từ đây
)

# Tạo generator cho dữ liệu huấn luyện và kiểm tra
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(img_size, img_size),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training"
)

validation_generator = validation_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(img_size, img_size),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)

'''# Xây dựng mô hình CNN
model = Sequential()

# Convolutional layers + Max Pooling + BatchNormalization
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_size, img_size, 1)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flattening và Dense layers
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))  # 7 lớp cho 7 cảm xúc

# Biên dịch mô hình
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Cài đặt callback để lưu mô hình tốt nhất
checkpoint = ModelCheckpoint(
    'model_best.keras', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max'
)

# Huấn luyện mô hình
epochs = 60
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

# Vẽ biểu đồ độ chính xác và độ mất mát
fig, ax = plt.subplots(1, 2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
fig.set_size_inches(12, 4)

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Training Accuracy vs Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Training Loss vs Validation Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Đánh giá mô hình trên dữ liệu huấn luyện và kiểm tra
train_loss, train_acc = model.evaluate(train_generator)
test_loss, test_acc = model.evaluate(validation_generator)

print(f"Final Train Accuracy = {train_acc*100:.2f}%, Validation Accuracy = {test_acc*100:.2f}%")

# Lưu mô hình sau khi huấn luyện
model.save('model_final.h5')
model.save_weights('model_weights_final.weights.h5')'''

# Tải mô hình đã huấn luyện
model = load_model('model_best.keras')  # hoặc 'model_final.h5' nếu bạn muốn tải mô hình cuối cùng

# Đọc ảnh thử (đảm bảo là ảnh phù hợp với kích thước và kiểu màu mà mô hình yêu cầu)
img_path = 'sp.png'

# Tải ảnh thử và thay đổi kích thước về (48, 48), grayscale
img = image.load_img(img_path, target_size=(img_size, img_size), color_mode="grayscale")

# Chuyển ảnh thành mảng numpy và chuẩn hóa giá trị pixel
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Thêm một chiều batch
img_array = img_array / 255.0  # Chuẩn hóa giá trị pixel (giống như trong quá trình huấn luyện)

# Dự đoán cảm xúc của ảnh thử
prediction = model.predict(img_array)

# Hiển thị kết quả dự đoán
class_labels = train_generator.class_indices  # Dễ dàng lấy tên lớp từ generator
class_labels = {v: k for k, v in class_labels.items()}  # Đảo ngược từ điển để lấy tên từ chỉ số lớp
predicted_class = np.argmax(prediction, axis=1)[0]

print(f"Dự đoán cảm xúc là: Đẹp trai mỗi tội không có bồ")
