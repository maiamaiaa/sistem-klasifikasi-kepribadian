import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def load_and_prepare_model():
    try:
        # Load data
        df = pd.read_csv("personality_dataset.csv")
        df = df.dropna()
        
        # Encode data
        le_stage = LabelEncoder()
        le_drained = LabelEncoder()
        
        df['Stage_fear'] = le_stage.fit_transform(df['Stage_fear'])
        df['Drained_after_socializing'] = le_drained.fit_transform(df['Drained_after_socializing'])
        df['Personality'] = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})
        
        # Pisahkan fitur dan target
        X = df.drop(columns='Personality')
        y = df['Personality']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalisasi data (penting untuk SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Latih model SVM
        svm = SVC(
            kernel='rbf',           # Radial Basis Function kernel
            C=1.0,                  # Regularization parameter
            gamma='scale',          # Kernel coefficient
            probability=True,       # Enable probability estimates
            random_state=42
        )
        svm.fit(X_train_scaled, y_train)
        
        # Evaluasi model
        y_pred = svm.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model SVM berhasil dilatih dengan akurasi: {accuracy:.2%}")
        print(f"Kernel yang digunakan: {svm.kernel}")
        print(f"Jumlah Support Vectors: {svm.n_support_}")
        print("-" * 50)
        
        return svm, scaler, X.columns
        
    except FileNotFoundError:
        print("Error: File 'personality_dataset.csv' tidak ditemukan!")
        print("Pastikan file dataset ada di direktori yang sama dengan script ini.")
        return None, None, None
    except Exception as e:
        print(f"Error saat memuat data: {e}")
        return None, None, None

def get_user_input():
    """Mengumpulkan input dari user secara interaktif"""
    print("=== PREDIKSI KEPRIBADIAN DENGAN SVM ===")
    print("Silakan jawab pertanyaan berikut untuk memprediksi tipe kepribadian Anda:")
    print()
    
    # Pertanyaan 1: Time spent alone
    while True:
        try:
            time_alone = float(input("1. Berapa jam Anda biasanya menghabiskan waktu sendirian per hari? (0-24): "))
            if 0 <= time_alone <= 24:
                break
            else:
                print("   Masukkan angka antara 0-24 jam.")
        except ValueError:
            print("   Masukkan angka yang valid.")
    
    # Pertanyaan 2: Stage fear
    while True:
        stage_fear = input("2. Apakah Anda takut tampil di depan umum? (ya/tidak): ").lower().strip()
        if stage_fear in ['ya', 'tidak', 'yes', 'no']:
            stage_fear_encoded = 1 if stage_fear in ['ya', 'yes'] else 0
            break
        else:
            print("   Jawab dengan 'ya' atau 'tidak'.")
    
    # Pertanyaan 3: Social event attendance
    while True:
        try:
            social_events = int(input("3. Berapa kali Anda menghadiri acara sosial per bulan? (0-30): "))
            if 0 <= social_events <= 30:
                break
            else:
                print("   Masukkan angka antara 0-30.")
        except ValueError:
            print("   Masukkan angka yang valid.")
    
    # Pertanyaan 4: Going outside
    while True:
        try:
            going_outside = int(input("4. Berapa kali Anda keluar rumah untuk bersosialisasi per minggu? (0-20): "))
            if 0 <= going_outside <= 20:
                break
            else:
                print("   Masukkan angka antara 0-20.")
        except ValueError:
            print("   Masukkan angka yang valid.")
    
    # Pertanyaan 5: Drained after socializing
    while True:
        drained = input("5. Apakah Anda merasa lelah setelah bersosialisasi? (ya/tidak): ").lower().strip()
        if drained in ['ya', 'tidak', 'yes', 'no']:
            drained_encoded = 1 if drained in ['ya', 'yes'] else 0
            break
        else:
            print("   Jawab dengan 'ya' atau 'tidak'.")
    
    # Pertanyaan 6: Friends circle size
    while True:
        try:
            friends_circle = int(input("6. Berapa banyak teman dekat yang Anda miliki? (0-50): "))
            if 0 <= friends_circle <= 50:
                break
            else:
                print("   Masukkan angka antara 0-50.")
        except ValueError:
            print("   Masukkan angka yang valid.")
    
    # Pertanyaan 7: Post frequency
    while True:
        try:
            post_freq = int(input("7. Berapa kali Anda posting di media sosial per minggu? (0-50): "))
            if 0 <= post_freq <= 50:
                break
            else:
                print("   Masukkan angka antara 0-50.")
        except ValueError:
            print("   Masukkan angka yang valid.")
    
    return {
        'Time_spent_Alone': time_alone,
        'Stage_fear': stage_fear_encoded,
        'Social_event_attendance': social_events,
        'Going_outside': going_outside,
        'Drained_after_socializing': drained_encoded,
        'Friends_circle_size': friends_circle,
        'Post_frequency': post_freq
    }

def predict_personality(model, scaler, feature_columns, user_data):
    """Memprediksi kepribadian berdasarkan input user"""
    input_df = pd.DataFrame([user_data], columns=feature_columns)
    
    # Normalisasi input dengan scaler yang sama
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    personality_type = "Extrovert" if prediction == 1 else "Introvert"
    confidence = max(probability) * 100
    
    # Hitung jarak ke hyperplane (decision boundary)
    decision_score = model.decision_function(input_scaled)[0]
    
    return personality_type, confidence, decision_score

def get_svm_insights(model, scaler, feature_columns, user_data):
    """Mendapatkan insight dari model SVM"""
    input_df = pd.DataFrame([user_data], columns=feature_columns)
    input_scaled = scaler.transform(input_df)
    
    # Jarak ke hyperplane
    decision_score = model.decision_function(input_scaled)[0]
    
    # Interpretasi jarak
    if abs(decision_score) > 1.0:
        confidence_level = "Sangat Yakin"
    elif abs(decision_score) > 0.5:
        confidence_level = "Cukup Yakin"
    else:
        confidence_level = "Kurang Yakin"
    
    return {
        'decision_score': decision_score,
        'confidence_level': confidence_level,
        'support_vectors': model.n_support_
    }

def show_results(personality_type, confidence, decision_score, user_data, svm_insights):
    """Menampilkan hasil prediksi dengan format yang menarik"""
    print("\n" + "="*60)
    print("                HASIL PREDIKSI KEPRIBADIAN (SVM)")
    print("="*60)
    print(f"🎯 Tipe Kepribadian Anda: {personality_type.upper()}")
    print(f"📊 Tingkat Keyakinan: {confidence:.1f}%")
    print(f"🔍 Tingkat Kepercayaan Model: {svm_insights['confidence_level']}")
    print("="*60)
    
    if personality_type == "Introvert":
        print("📝 KARAKTERISTIK INTROVERT:")
        print("   • Lebih nyaman dengan waktu sendirian")
        print("   • Cenderung memiliki lingkaran pertemanan yang kecil tapi dekat")
        print("   • Membutuhkan waktu untuk 'recharge' setelah bersosialisasi")
        print("   • Lebih suka aktivitas yang tenang dan mendalam")
    else:
        print("📝 KARAKTERISTIK EXTROVERT:")
        print("   • Energik dalam situasi sosial")
        print("   • Senang bertemu orang baru dan berbagi pengalaman")
        print("   • Cenderung ekspresif dan terbuka")
        print("   • Mendapatkan energi dari interaksi dengan orang lain")
    
    print("\n🤖 INFORMASI TEKNIS SVM:")
    print(f"   • Decision Score: {decision_score:.3f}")
    if decision_score > 0:
        print("   • Posisi: Di sisi Extrovert dari hyperplane")
    else:
        print("   • Posisi: Di sisi Introvert dari hyperplane")
    print(f"   • Jarak ke Decision Boundary: {abs(decision_score):.3f}")
    print(f"   • Support Vectors: {svm_insights['support_vectors'][0]} (Introvert), {svm_insights['support_vectors'][1]} (Extrovert)")
    
    print("\n📋 RINGKASAN JAWABAN ANDA:")
    print(f"   • Waktu sendirian per hari: {user_data['Time_spent_Alone']} jam")
    print(f"   • Takut tampil di depan umum: {'Ya' if user_data['Stage_fear'] == 1 else 'Tidak'}")
    print(f"   • Acara sosial per bulan: {user_data['Social_event_attendance']} kali")
    print(f"   • Keluar untuk bersosialisasi per minggu: {user_data['Going_outside']} kali")
    print(f"   • Lelah setelah bersosialisasi: {'Ya' if user_data['Drained_after_socializing'] == 1 else 'Tidak'}")
    print(f"   • Jumlah teman dekat: {user_data['Friends_circle_size']} orang")
    print(f"   • Posting media sosial per minggu: {user_data['Post_frequency']} kali")
    print("="*60)



def main():
    """Fungsi utama program"""
    print("Memuat model prediksi kepribadian dengan SVM...")
    
    model, scaler, feature_columns = load_and_prepare_model()
    
    if model is None:
        return
    
    while True:
        try:
            user_data = get_user_input()
            
            personality_type, confidence, decision_score = predict_personality(
                model, scaler, feature_columns, user_data
            )
            
            svm_insights = get_svm_insights(model, scaler, feature_columns, user_data)
            
            show_results(personality_type, confidence, decision_score, user_data, svm_insights)
            
            print("\nApakah Anda ingin melakukan prediksi lagi? (ya/tidak): ", end="")
            lagi = input().lower().strip()
            
            if lagi not in ['ya', 'yes', 'y']:
                print("\nTerima kasih telah menggunakan program prediksi kepribadian!")
                print("Semoga hasil prediksi ini bermanfaat untuk Anda. 😊")
                break
            else:
                print("\n" + "="*60)
                
        except KeyboardInterrupt:
            print("\n\nProgram dihentikan oleh user. Terima kasih!")
            break
        except Exception as e:
            print(f"\nTerjadi error: {e}")
            print("Silakan coba lagi.")

if __name__ == "__main__":
    main()