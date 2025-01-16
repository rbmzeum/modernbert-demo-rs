use embed_anything::embeddings::local::text_embedding::ONNXModel;
use embed_anything::embeddings::embed::Embedder;
use embed_anything::Dtype;
use tokio; // Используем tokio как асинхронную среду выполнения

#[tokio::main] // Включаем асинхронный контекст
async fn main() {
    // Загрузка предобученной модели BERT
    let model = Embedder::from_pretrained_onnx(
        "bert", // model_architecture: &str
        Some(ONNXModel::ModernBERTBase), // model_name: Option<ONNXModel>
        None, // model_id: Option<&str>
        None, // revision: Option<&str>
        Some(Dtype::Q4F16), // dtype: Option<Dtype>
        None, // path_in_repo: Option<&str>
    ).expect("Не удалось загрузить модель");

    // Текст, который нужно преобразовать в вектор
    let texts = vec![
        "Пример текста для преобразования в вектор.".to_string(),
    ];

    // Преобразование текста в вектор
    let embeddings = model
        .embed(&texts, None) // batch_size: None (используем значение по умолчанию)
        .await
        .expect("Не удалось преобразовать текст в вектор");

    // Вывод результата
    println!("Векторное представление текста: {:?}", embeddings);
}