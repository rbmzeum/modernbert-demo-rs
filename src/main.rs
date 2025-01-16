use embed_anything::embeddings::local::text_embedding::ONNXModel;
use embed_anything::embeddings::embed::{Embedder, EmbeddingResult};
use embed_anything::Dtype;
use tokio; // Используем tokio как асинхронную среду выполнения
use ndarray::Array1; // Для работы с векторами
use prettytable::{Table, row}; // Для вывода таблицы

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

    // Списки текстов из разных семантик
    let semantic_list_1 = vec![
        "Собака играет в парке.".to_string(),
        "Щенок гоняется за мячом.".to_string(),
        "Собака лает на прохожих.".to_string(),
    ];

    let semantic_list_2 = vec![
        "Кот спит на диване.".to_string(),
        "Котенок играет с клубком ниток.".to_string(),
        "Кот мурлычет на солнце.".to_string(),
    ];

    // Нейтральные проверочные фразы
    let test_phrase_1 = "Собака радостно бегает по траве.".to_string(); // Близка к первой семантике
    let test_phrase_2 = "Кот нежится на подоконнике.".to_string(); // Близка ко второй семантике

    // Объединяем все тексты для получения их векторных представлений
    let all_texts = [&semantic_list_1[..], &semantic_list_2[..], &[test_phrase_1.clone(), test_phrase_2.clone()]].concat();

    // Преобразуем тексты в векторы
    let embeddings = model
        .embed(&all_texts, None) // batch_size: None (используем значение по умолчанию)
        .await
        .expect("Не удалось преобразовать текст в вектор");

    // Разделяем векторы на списки и проверочные фразы
    let semantic_embeddings_1 = &embeddings[0..semantic_list_1.len()];
    let semantic_embeddings_2 = &embeddings[semantic_list_1.len()..semantic_list_1.len() + semantic_list_2.len()];
    let test_embedding_1 = &embeddings[semantic_list_1.len() + semantic_list_2.len()];
    let test_embedding_2 = &embeddings[semantic_list_1.len() + semantic_list_2.len() + 1];

    // Функция для вычисления евклидова расстояния между двумя векторами
    fn euclidean_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        ((a - b).mapv(|x| x.powi(2)).sum()).sqrt()
    }

    // Функция для извлечения вектора из EmbeddingResult
    fn extract_vector(embedding_result: &EmbeddingResult) -> Array1<f32> {
        match embedding_result {
            EmbeddingResult::DenseVector(data) => Array1::from_vec(data.clone()),
            EmbeddingResult::MultiVector(data) => {
                // Если MultiVector, выбираем первый вектор (или объединяем их)
                Array1::from_vec(data[0].clone())
            }
        }
    }

    // Преобразуем векторы в ndarray::Array1
    let test_embedding_1 = extract_vector(test_embedding_1);
    let test_embedding_2 = extract_vector(test_embedding_2);

    // Вычисляем расстояния для test_phrase_1
    let mut distances_1 = Vec::new();
    for embedding in semantic_embeddings_1.iter().chain(semantic_embeddings_2.iter()) {
        let embedding = extract_vector(embedding);
        distances_1.push(euclidean_distance(&test_embedding_1, &embedding));
    }

    // Вычисляем расстояния для test_phrase_2
    let mut distances_2 = Vec::new();
    for embedding in semantic_embeddings_1.iter().chain(semantic_embeddings_2.iter()) {
        let embedding = extract_vector(embedding);
        distances_2.push(euclidean_distance(&test_embedding_2, &embedding));
    }

    // Создаём таблицу для вывода результатов
    let mut table = Table::new();

    // Заголовки таблицы
    table.add_row(row!["Текст", "Расстояние до test_phrase_1", "Расстояние до test_phrase_2"]);

    // Добавляем строки с результатами для первой семантики
    for (i, text) in semantic_list_1.iter().enumerate() {
        table.add_row(row![text, distances_1[i], distances_2[i]]);
    }

    // Добавляем строки с результатами для второй семантики
    for (i, text) in semantic_list_2.iter().enumerate() {
        table.add_row(row![text, distances_1[semantic_list_1.len() + i], distances_2[semantic_list_1.len() + i]]);
    }

    // Выводим таблицу
    table.printstd();
}