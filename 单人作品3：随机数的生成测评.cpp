#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>


#define MAX_INTERVAL 10000

void count_frequency(){//count the frequency of random num(ues the way"x=rand()%n;")
	printf("������N��");
	int N,count_times;
	scanf("%d",&N);
	printf("������������������������������)��");
	scanf("%d",&count_times); 
	printf("\n");
	printf("���������ɵ�������У�\n");
	int times[N]={0};
	int i,x;
	srand(time(NULL));
	for(i=0;i<count_times;i++){
		x=rand()%N;printf("%d\t",x);
		times[x]++;
	}
    printf("\n");
	float p[N]={0};
	printf("ͳ��ÿ�����ĳ��ָ���p(i)������£�\n;");
	for(i=0;i<N;i++){
		p[i]=float(times[i])/float(count_times);printf("p(%d)=%f\t",i,p[i]);
	}
}

void count_repetition_intervals() {//ͳ�Ƽ���ֲ���� 
    int N, count_times;
    
    // ����N�Ͳ��Դ���
    printf("������N��ֵ(0��N-1�ķ�Χ): ");
    if (scanf("%d", &N) != 1 || N <= 0) {
        printf("����: N������������!\n");
        return;
    }
    
    printf("��������Դ���(�������������): ");
    if (scanf("%d", &count_times) != 1 || count_times <= 0) {
        printf("����: ���Դ���������������!\n");
        return;
    }
    
    // ��ʼ�����������
    int intervals[N][MAX_INTERVAL] = {0}; // intervals[i][k]��ʾ����i���k���ֵĴ���
    int last_pos[N];                     // ��¼ÿ�������ϴγ��ֵ�λ��
    for (int i = 0; i < N; i++) {
        last_pos[i] = -1;                // ��ʼ��Ϊ-1��ʾ��δ����
    }
    
    // �����������ͳ�Ƽ��
    srand(time(NULL));
    for (int i = 0; i < count_times; i++) {
        int x = rand() % N;printf("%d\t",x);
        
        // ������ǵ�һ�γ��֣���¼���
        if (last_pos[x] != -1) {
            int gap = i - last_pos[x] - 1;
            if (gap < MAX_INTERVAL) {
                intervals[x][gap]++;
            }
        }
        
        // �����ϴγ���λ��
        last_pos[x] = i;
    }
    
    // ��ӡÿ�����ֵļ���ֲ�
    printf("\n=== �����ظ�����ֲ�ͳ�� ===\n");
    printf("N = %d, ���Դ��� = %d\n\n", N, count_times);
    
    for (int i = 0; i < N; i++) {
        printf("���� %d �ļ���ֲ�:\n", i);
        printf("���\t���ִ���\tƵ��\n");
        
        int total_gaps = 0;
        for (int k = 0; k < MAX_INTERVAL; k++) {
            total_gaps += intervals[i][k];
        }
        
        // ֻ��ӡ���ִ�������ļ��
        int printed = 0;
        for (int k = 0; k < MAX_INTERVAL; k++) {
            if (intervals[i][k] > 0) {
                double frequency = (double)intervals[i][k] / total_gaps;
                printf("%d\t%d\t\t%.6f\n", k, intervals[i][k], frequency);
                printed++;
                
                // ��������������������
                if (printed >= 20 && k < MAX_INTERVAL - 1) {
                    printf("... (ʡ��������Ƶ�ʼ��)\n");
                    break;
                }
            }
        }
        
        if (printed == 0) {
            printf("���ظ����ֵ����\n");
        }
        
        printf("\n");
    }
}
#define MAX_PRINT 100  // ����ӡ�����������

void entropy_test() {//��ֵ���� 
    int N, count_times;
    
    // ����N�Ͳ��Դ���
    printf("������N��ֵ(0��N-1�ķ�Χ): ");
    if (scanf("%d", &N) != 1 || N <= 0) {
        printf("����: N������������!\n");
        return;
    }
    
    printf("�������������(���������������): ");
    if (scanf("%d", &count_times) != 1 || count_times <= 0) {
        printf("����: ��������������������!\n");
        return;
    }
    
    // ��ʼ�������������
    srand(time(NULL));
    
    // �����������ͳ��Ƶ��
    int samples[count_times];
    int counts[MAX_PRINT] = {0};  // ��¼ÿ����ֵ�ĳ��ִ���
    
    printf("\n���ɵ����������(�����ʾǰ%d��):\n", MAX_PRINT);
    for (int i = 0; i < count_times; i++) {
        samples[i] = rand() % N;
        counts[samples[i]]++;
        
        // ��ӡǰMAX_PRINT�������
        if (i < MAX_PRINT) {
            printf("%d ", samples[i]);
            if ((i + 1) % 10 == 0) printf("\n");  // ÿ����ʾ10����
        } else if (i == MAX_PRINT) {
            printf("... (��%d�������)\n", count_times);
        }
    }
    
    // ����ʵ����ֵ
    double entropy = 0.0;
    for (int i = 0; i < N; i++) {
        if (counts[i] > 0) {
            double p = (double)counts[i] / count_times;
            entropy -= p * log2(p);  // ��Ϣ�ع�ʽ
        }
    }
    
    // ����������ֵ�����ȷֲ����أ�
    double ideal_entropy = log2(N);
    
    // �жϲ����Ƿ�ͨ�����趨��ֵΪ�����ص�95%��
    double threshold = ideal_entropy * 0.95;
    int passed = (entropy >= threshold);
    
    // ��ӡ���Խ��
    printf("\n=== �ز��Խ�� ===\n");
    printf("N = %d, �������� = %d\n", N, count_times);
    printf("ʵ����ֵ: %.6f ����/����\n", entropy);
    printf("������ֵ: %.6f ����/���� (���ȷֲ����������ֵ)\n", ideal_entropy);
    printf("ͨ����׼: ʵ���� �� ������ �� 95%%\n");
    printf("���Խ��: %s\n", passed ? "ͨ��" : "δͨ��");
    printf("��Ч��: %.2f%%\n", (entropy / ideal_entropy) * 100);
}

void runs_test() {//�γ̲��� 
    int N, count_times;
    
    // ����N�Ͳ��Դ���
    printf("������N��ֵ(0��N-1�ķ�Χ): ");
    if (scanf("%d", &N) != 1 || N <= 0) {
        printf("����: N������������!\n");
        return;
    }
    
    printf("�������������(���������������): ");
    if (scanf("%d", &count_times) != 1 || count_times <= 0) {
        printf("����: ��������������������!\n");
        return;
    }
    
    // ��ʼ�������������
    srand(time(NULL));
    
    // �����������ת��Ϊ���������У�������λ����
    int samples[count_times];
    int binary[count_times];
    
    // ������λ�����򻯰棺ʹ��(N-1)/2��Ϊ������λ����
    double median = (N - 1) / 2.0;
    
    printf("\n���ɵ����������(�����ʾǰ%d��):\n", MAX_PRINT);
    for (int i = 0; i < count_times; i++) {
        samples[i] = rand() % N;
        binary[i] = (samples[i] >= median) ? 1 : 0;
        
        // ��ӡǰMAX_PRINT�������
        if (i < MAX_PRINT) {
            printf("%d ", samples[i]);
            if ((i + 1) % 10 == 0) printf("\n");  // ÿ����ʾ10����
        } else if (i == MAX_PRINT) {
            printf("... (��%d�������)\n", count_times);
        }
    }
    
    // �����γ�����Runs��
    int runs = 1;
    for (int i = 1; i < count_times; i++) {
        if (binary[i] != binary[i-1]) {
            runs++;
        }
    }
    
    // ����0��1������
    int n0 = 0, n1 = 0;
    for (int i = 0; i < count_times; i++) {
        if (binary[i] == 0) n0++;
        else n1++;
    }
    
    // �������������γ����ͱ�׼��
    double expected_runs = 2.0 * n0 * n1 / count_times + 0.5;
    double variance = 2.0 * n0 * n1 * (2.0 * n0 * n1 - count_times) / 
                     (count_times * count_times * (count_times - 1));
    double std_dev = sqrt(variance);
    
    // ����z-score����׼��̬�ֲ��µ�ͳ������
    double z_score = (runs - expected_runs) / std_dev;
    
    // �жϲ����Ƿ�ͨ����95%�������䣺|z| < 1.96��
    int passed = (fabs(z_score) < 1.96);
    
    // ��ӡ���Խ��
    printf("\n=== �γ̲��Խ�� ===\n");
    printf("N = %d, �������� = %d\n", N, count_times);
    printf("��λ����׼: %.1f\n", median);
    printf("0������: %d\n", n0);
    printf("1������: %d\n", n1);
    printf("ʵ���γ���: %d\n", runs);
    printf("���������γ���: %.2f\n", expected_runs);
    printf("��׼��: %.4f\n", std_dev);
    printf("z-score: %.4f\n", z_score);
    printf("ͨ����׼: |z-score| < 1.96 (95%%��������)\n");
    printf("���Խ��: %s\n", passed ? "ͨ��" : "δͨ��");
}

void autocorrelation_test() {//������Բ��� 
    int N, count_times;
    
    // ����N�Ͳ��Դ���
    printf("������N��ֵ(0��N-1�ķ�Χ): ");
    if (scanf("%d", &N) != 1 || N <= 0) {
        printf("����: N������������!\n");
        return;
    }
    
    printf("�������������(���������������): ");
    if (scanf("%d", &count_times) != 1 || count_times <= 0) {
        printf("����: ��������������������!\n");
        return;
    }
    
    // ��ʼ�������������
    srand(time(NULL));
    
    // �������������
    int samples[count_times];
    
    printf("\n���ɵ����������(�����ʾǰ%d��):\n", MAX_PRINT);
    for (int i = 0; i < count_times; i++) {
        samples[i] = rand() % N;
        
        // ��ӡǰMAX_PRINT�������
        if (i < MAX_PRINT) {
            printf("%d ", samples[i]);
            if ((i + 1) % 10 == 0) printf("\n");  // ÿ����ʾ10����
        } else if (i == MAX_PRINT) {
            printf("... (��%d�������)\n", count_times);
        }
    }
    
    // �������о�ֵ
    double mean = 0.0;
    for (int i = 0; i < count_times; i++) {
        mean += samples[i];
    }
    mean /= count_times;
    
    // ���������ϵ�����ͺ�1��
    double numerator = 0.0;
    double denominator = 0.0;
    
    for (int i = 0; i < count_times - 1; i++) {
        numerator += (samples[i] - mean) * (samples[i+1] - mean);
        denominator += pow(samples[i] - mean, 2);
    }
    
    // �������һ�������ķ�ĸ����
    denominator += pow(samples[count_times-1] - mean, 2);
    
    double autocorrelation = numerator / denominator;
    
    // �����׼������ֵ������������У���׼���ԼΪ1/��n��
    double standard_error = 1.0 / sqrt(count_times);
    
    // ����z-score�������ϵ����0��ƫ��̶ȣ�
    double z_score = autocorrelation / standard_error;
    
    // �жϲ����Ƿ�ͨ����95%�������䣺|z| < 1.96��
    int passed = (fabs(z_score) < 1.96);
    
    // ��ӡ���Խ��
    printf("\n=== ������Բ��Խ�� ===\n");
    printf("N = %d, �������� = %d\n", N, count_times);
    printf("���о�ֵ: %.4f\n", mean);
    printf("��������ֵ: 0 (��ȫ������е������ϵ��Ӧ�ӽ�0)\n");
    printf("�ͺ�1�������ϵ��: %.6f\n", autocorrelation);
    printf("��׼���: %.6f\n", standard_error);
    printf("z-score: %.4f\n", z_score);
    printf("ͨ����׼: |z-score| < 1.96 (95%%��������)\n");
    printf("���Խ��: %s\n", passed ? "ͨ��" : "δͨ��");
}
double calculate_chi_square_p_value(double chi_square, int df);
void frequency_test() {
    int N, count_times;
    
    // ����N�Ͳ��Դ���
    printf("������N��ֵ(0��N-1�ķ�Χ): ");
    if (scanf("%d", &N) != 1 || N <= 0) {
        printf("����: N������������!\n");
        while (getchar() != '\n'); // ������뻺����
        return;
    }
    
    printf("�������������(���������������): ");
    if (scanf("%d", &count_times) != 1 || count_times <= 0) {
        printf("����: ��������������������!\n");
        while (getchar() != '\n'); // ������뻺����
        return;
    }
    
    // ��̬�����ڴ�
    int *samples = (int *)malloc(count_times * sizeof(int));
    int *counts = (int *)calloc(N, sizeof(int)); // ʹ��N��Ϊ��С
    
    if (!samples || !counts) {
        printf("�ڴ����ʧ��!\n");
        if (samples) free(samples);
        return;
    }
    
    // �����������ͳ��Ƶ��
    printf("\n���ɵ����������(�����ʾǰ%d��):\n", MAX_PRINT);
    for (int i = 0; i < count_times; i++) {
        samples[i] = rand() % N;
        counts[samples[i]]++;
        
        // ��ӡǰMAX_PRINT�������
        if (i < MAX_PRINT) {
            printf("%d ", samples[i]);
            if ((i + 1) % 10 == 0) printf("\n");
        } else if (i == MAX_PRINT) {
            printf("... (��%d�������)\n", count_times);
        }
    }
    
    // ���㿨��ͳ����
    double expected = (double)count_times / N;
    double chi_square = 0.0;
    
    printf("\n=== Ƶ�ʷֲ�ͳ�� ===\n");
    printf("��ֵ\tʵ�ʴ���\t��������\tƫ��\n");
    for (int i = 0; i < N; i++) {
        double deviation = counts[i] - expected;
        chi_square += pow(deviation, 2) / expected;
        printf("%d\t%d\t\t%.2f\t\t%.2f\n", i, counts[i], expected, deviation);
    }
    
    // �������ɶ�
    int degrees_of_freedom = N - 1;
    
    // ʹ�ø�׼ȷ��pֵ���㷽��
    double p_value = calculate_chi_square_p_value(chi_square, degrees_of_freedom);
    
    // �жϲ����Ƿ�ͨ��
    int passed = (p_value > 0.05);
    
    // ��ӡ���Խ��
    printf("\n=== Ƶ�ʷֲ����Խ�� ===\n");
    printf("N = %d, �������� = %d\n", N, count_times);
    printf("����ͳ����(Chi-Square): %.4f\n", chi_square);
    printf("���ɶ�: %d\n", degrees_of_freedom);
    printf("pֵ: %.6f\n", p_value);
    printf("ͨ����׼: pֵ > 0.05 (�ֲ�����ȼ�������������)\n");
    printf("���Խ��: %s\n", passed ? "ͨ��" : "δͨ��");
    
    // �ͷ��ڴ�
    free(samples);
    free(counts);
}

// ��Ҫ��ӿ����ֲ�pֵ���㺯��
double calculate_chi_square_p_value(double chi_square, int df) {
    // ʹ��Gamma���������׼ȷ��pֵ
    // �������һ����ʵ�֣�ʵ��Ӧ����Ӧ��ʹ����ѧ��
    if (df == 2) {
        return exp(-chi_square / 2);
    } else {
        // �򻯽��ƣ�ʵ��Ӧ����Ӧ��ʹ�ø���ȷ�ļ��㷽��
        double p = 0.5 * erfc(sqrt(chi_square / 2) - sqrt(df / 2 - 1));
        return p;
    }
}

void cumulative_sums_test() {//�ۼӲ��� 
    int N, count_times;
    
    // ����N�Ͳ��Դ���
    printf("������N��ֵ(0��N-1�ķ�Χ): ");
    if (scanf("%d", &N) != 1 || N <= 0) {
        printf("����: N������������!\n");
        return;
    }
    
    printf("�������������(�����������������): ");
    if (scanf("%d", &count_times) != 1 || count_times <= 0) {
        printf("����: ��������������������!\n");
        return;
    }
    
    // ��ʼ�������������
    srand(time(NULL));
    
    // ������������в�ת��Ϊ��׼��ֵ (-1 �� +1)
    int samples[count_times];
    double normalized[count_times];
    
    printf("\n���ɵ����������(�����ʾǰ%d��):\n", MAX_PRINT);
    for (int i = 0; i < count_times; i++) {
        samples[i] = rand() % N;
        // �������ת��Ϊ -1 �� +1 (�����Ƿ���ڵ�����λ��)
        normalized[i] = (samples[i] >= (N-1)/2.0) ? 1.0 : -1.0;
        
        // ��ӡǰMAX_PRINT�������
        if (i < MAX_PRINT) {
            printf("%d ", samples[i]);
            if ((i + 1) % 10 == 0) printf("\n");  // ÿ����ʾ10����
        } else if (i == MAX_PRINT) {
            printf("... (��%d�������)\n", count_times);
        }
    }
    
    // ���������ۼӺ� (Cusum Forward)
    double max_forward = 0.0;
    double sum = 0.0;
    for (int i = 0; i < count_times; i++) {
        sum += normalized[i];
        if (fabs(sum) > max_forward) {
            max_forward = fabs(sum);
        }
    }
    
    // ���㷴���ۼӺ� (Cusum Reverse)
    double max_reverse = 0.0;
    sum = 0.0;
    for (int i = count_times-1; i >= 0; i--) {
        sum += normalized[i];
        if (fabs(sum) > max_reverse) {
            max_reverse = fabs(sum);
        }
    }
    
    // ����ͳ���� (��׼���������ۼӺ�)
    double z_forward = max_forward / sqrt(count_times);
    double z_reverse = max_reverse / sqrt(count_times);
    
    // ����pֵ (ʹ�ý��ƹ�ʽ)
    double p_forward = 1.0;
    double p_reverse = 1.0;
    
    // ����z > 1.0��ʹ�ý��ƹ�ʽ
    if (z_forward > 1.0) {
        p_forward = 2.0 * (1.0 - 0.5 * exp(-1.7725 * z_forward * z_forward));
    }
    
    if (z_reverse > 1.0) {
        p_reverse = 2.0 * (1.0 - 0.5 * exp(-1.7725 * z_reverse * z_reverse));
    }
    
    // �ۺ�pֵ (ȡ��Сֵ)
    double p_value = (p_forward < p_reverse) ? p_forward : p_reverse;
    
    // �жϲ����Ƿ�ͨ�� (pֵ����0.01��ʾͨ��)
    int passed = (p_value > 0.01);
    
    // ��ӡ���Խ��
    printf("\n=== �ۼӺͲ��Խ�� ===\n");
    printf("N = %d, �������� = %d\n", N, count_times);
    printf("��������ۼӺ�: %.6f\n", max_forward);
    printf("��������ۼӺ�: %.6f\n", max_reverse);
    printf("��׼��ͳ���� (Z): ����=%.6f, ����=%.6f\n", z_forward, z_reverse);
    printf("pֵ: %.6f\n", p_value);
    printf("ͨ����׼: pֵ > 0.01 (�������������)\n");
    printf("���Խ��: %s\n", passed ? "ͨ��" : "δͨ��");
}

// ����0��N-1֮��ľ��ȷֲ����������
double uniform_random(double N) {
    return (double)rand() / RAND_MAX * N;
}

// ʹ��Box-Muller�任������̬�ֲ������
double normal_random(double mean, double stddev) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return mean + stddev * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// ��������ľ�ֵ
double mean(double arr[], int n) {
    double sum = 0.0;
    int valid_count = 0;
    for (int i = 0; i < n; i++) {
        if (!isnan(arr[i]) && !isinf(arr[i])) {
            sum += arr[i];
            valid_count++;
        }
    }
    if (valid_count == 0) {
        return 0.0; // ���������
    }
    return sum / valid_count;
}

// ��������ı�׼��
double stddev(double arr[], int n, double mean_val) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        sum_sq += (arr[i] - mean_val) * (arr[i] - mean_val);
    }
    return sqrt(sum_sq / (n - 1));
}

// �����������λ�������޸�ԭ���飩
double median(double arr[], int n) {
    // ð������
    double temp;
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
    
    if (n % 2 == 0)
        return (arr[n/2 - 1] + arr[n/2]) / 2.0;
    else
        return arr[n/2];
}

// Lilliefors�������ϸʵ�֣����ؼ���ͳ������pֵ
double lilliefors_test(double arr[], int n, double *p_value) {
    // 1. ���Ʋ���������
    double sorted_data[10000];
    for (int i = 0; i < n; i++) {
        sorted_data[i] = arr[i];
    }
    
    // ð������
    double temp;
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (sorted_data[j] > sorted_data[j + 1]) {
                temp = sorted_data[j];
                sorted_data[j] = sorted_data[j + 1];
                sorted_data[j + 1] = temp;
            }
        }
    }
double mean(double arr[], int n);
 
    // 2. ����������ֵ�ͱ�׼��
    double sample_mean = mean(sorted_data, n);
    double sample_std = stddev(sorted_data, n, sample_mean);
    
    printf("\nLilliefors����������:\n");
    printf("------------------------\n");
    printf("������ֵ: %.4f\n", sample_mean);
    printf("������׼��: %.4f\n", sample_std);
    
    // 3. ���㾭���ۻ��ֲ�����(ECDF)������CDF֮���������
    double max_diff = 0.0;
    int max_idx = 0;
    
    // ���ǰ10����ļ������
    printf("\nǰ10�����ݵ��ECDF������CDF�Ƚ�:\n");
    printf("   ���ݵ� | ��׼��ֵ | ����CDF   | ECDF_lower | ECDF_upper | ����_lower | ����_upper\n");
    printf("----------|----------|-----------|------------|------------|------------|------------\n");
    
    for (int i = 0; i < n; i++) {
        double z = (sorted_data[i] - sample_mean) / sample_std;
        // ��׼��̬�ֲ���CDF�Ľ��Ƽ���
        double phi = 0.5 * (1.0 + erf(z / sqrt(2.0)));
        
        double ecdf_lower = (double)i / n;
        double ecdf_upper = (double)(i + 1) / n;
        
        double diff_lower = fabs(phi - ecdf_lower);
        double diff_upper = fabs(phi - ecdf_upper);
        
        // ����������
        if (diff_lower > max_diff) {
            max_diff = diff_lower;
            max_idx = i;
        }
        if (diff_upper > max_diff) {
            max_diff = diff_upper;
            max_idx = i;
        }
        
        // ���ǰ10����
        if (i < 10) {
            printf("%9d | %8.4f | %9.4f | %10.4f | %10.4f | %10.4f | %10.4f\n", 
                   i+1, z, phi, ecdf_lower, ecdf_upper, diff_lower, diff_upper);
        }
    }
    
    // ����������
    double z_max = (sorted_data[max_idx] - sample_mean) / sample_std;
    double phi_max = 0.5 * (1.0 + erf(z_max / sqrt(2.0)));
    double ecdf_lower_max = (double)max_idx / n;
    double ecdf_upper_max = (double)(max_idx + 1) / n;
    
    printf("\n�������: ���ݵ� #%d (ֵ=%.4f, ��׼��ֵ=%.4f)\n", 
           max_idx+1, sorted_data[max_idx], z_max);
    printf("����CDF: %.4f\n", phi_max);
    printf("����CDF��Χ: [%.4f, %.4f]\n", ecdf_lower_max, ecdf_upper_max);
    printf("Lilliefors����ͳ���� D = %.4f\n", max_diff);
    
    // ����pֵ���򻯷�����
    double critical_05 = 0.029 * sqrt(n); // 5%������ˮƽ�ٽ�ֵ����
    double critical_01 = 0.036 * sqrt(n); // 1%������ˮƽ�ٽ�ֵ����
    
    *p_value = (max_diff > critical_01) ? 0.01 : 
               (max_diff > critical_05) ? 0.05 : 
               0.10; // �򻯹���
    
    printf("�ٽ�ֵ (5%%): %.4f\n", critical_05);
    printf("����pֵ: %.4f\n", *p_value);
    
    return max_diff < critical_05;
}

// ���ȷֲ��Ŀ������飬������ϸ�������
double chi_square_test(double arr[], int n, double lower, double upper, int bins) {
    // ��ʼ��ֱ��ͼ����
    int observed[100] = {0};
    
    // ������������Ƶ��
    double bin_width = (upper - lower) / bins;
    double expected = (double)n / bins;
    
    printf("\n���ȷֲ���������������:\n");
    printf("------------------------\n");
    printf("������: %d\n", bins);
    printf("ÿ������Ƶ��: %.2f\n", expected);
    
    // ����ֱ��ͼ
    for (int i = 0; i < n; i++) {
        int bin = (arr[i] - lower) / bin_width;
        if (bin < 0) bin = 0;
        if (bin >= bins) bin = bins - 1;
        observed[bin]++;
    }
    
    // ���㿨��ͳ����
    double chi_square = 0.0;
    
    // ���ǰ10����ļ������
    printf("\nǰ10�����Ƶ���ֲ�:\n");
    printf(" ��� | ���䷶Χ       | �۲�Ƶ�� | ����Ƶ�� | (O-E)2/E\n");
    printf("------|----------------|----------|----------|---------\n");
    
    for (int i = 0; i < bins; i++) {
        double bin_start = lower + i * bin_width;
        double bin_end = bin_start + bin_width;
        double contribution = pow(observed[i] - expected, 2) / expected;
        chi_square += contribution;
        
        // ���ǰ10����
        if (i < 10) {
            printf("%5d | [%.2f, %.2f) | %8d | %8.2f | %7.4f\n", 
                   i+1, bin_start, bin_end, observed[i], expected, contribution);
        }
    }
    
    // ���ɶ� = bins - 1
    int df = bins - 1;
    
    // �����ٽ�ֵ���򻯲��
    double critical_05 = 0;
    if (df == 9) critical_05 = 16.919;  // ����ֵ
    else if (df == 19) critical_05 = 30.144;
    else critical_05 = df * 1.8;  // ����ֵ
    
    printf("\n����ͳ����: %.4f\n", chi_square);
    printf("���ɶ�: %d\n", df);
    printf("�ٽ�ֵ (5%%): %.4f\n", critical_05);
    
    // ����pֵ���򻯷�����
    double p_value = (chi_square > critical_05) ? 0.05 : 0.10;
    printf("����pֵ: %.4f\n", p_value);
    
    return chi_square < critical_05;
}

// ��ӡ��ǿ���ı�ֱ��ͼ
void print_enhanced_histogram(double arr[], int n, int bins, const char* title) {
    // �ҵ����ݷ�Χ
    double min = arr[0], max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] < min) min = arr[i];
        if (arr[i] > max) max = arr[i];
    }
    
    double range = max - min;
    double bin_width = range / bins;
    
    // ��ʼ��ֱ��ͼ����
    int hist[100] = {0};
    for (int i = 0; i < n; i++) {
        int bin = (arr[i] - min) / bin_width;
        if (bin >= bins) bin = bins - 1;
        hist[bin]++;
    }
    
    // �ҵ����Ƶ�������ڹ�һ��
    int max_count = 0;
    for (int i = 0; i < bins; i++) {
        if (hist[i] > max_count) max_count = hist[i];
    }
    
    // ��ӡֱ��ͼ����
    printf("\n%s (������=%d):\n", title, n);
    printf("��Χ: [%.2f, %.2f]\n", min, max);
    
    // ��ӡֱ��ͼ
    for (int i = 0; i < bins; i++) {
        double bin_start = min + i * bin_width;
        double bin_end = bin_start + bin_width;
        
        // �����Ǻ����������50����
        int stars = hist[i] * 50 / max_count;
        
        // ��ʽ����������Ƶ��
        printf("[%6.2f, %6.2f) | ", bin_start, bin_end);
        
        // �����Ǻ�
        for (int j = 0; j < stars; j++) {
            printf("*");
        }
        
        // ��ʾʵ��Ƶ��
        printf(" %d\n", hist[i]);
    }
}

void last_part_test() {
    double uniform_upper;  // ���ȷֲ�������
    double normal_mean;    // ��̬�ֲ��ľ�ֵ
    double normal_stddev;  // ��̬�ֲ��ı�׼��
    int count;             // ���ɵ����������
    int bins = 20;         // ֱ��ͼ������
    double p_value;        // ���ڴ洢����pֵ
    
    // ��ʼ�����������
    srand(time(NULL));
    printf("��������������Գ���\n");
    printf("====================\n");
    
    // ��ȡ�û�����
    printf("��������ȷֲ�������ֵ (����10.0): ");
    scanf("%lf", &uniform_upper);
    
    printf("������Ҫ���ɵ���������� (����1000): ");
    scanf("%d", &count);
    
    printf("��������̬�ֲ��ľ�ֵ (����0.0): ");
    scanf("%lf", &normal_mean);
    
    printf("��������̬�ֲ��ı�׼�� (����25.0): ");
    scanf("%lf", &normal_stddev);
    
    // ��֤����
    if (count <= 0 || count > 10000) {
        printf("�������������������1��10000֮��\n");
        return ;
    }
    
    if (uniform_upper <= 0) {
        printf("���󣺾��ȷֲ����ޱ������0\n");
        return ;
    }
    
    if (normal_stddev <= 0) {
        printf("������̬�ֲ���׼��������0\n");
        return ;
    }
    
    // ���ɾ��ȷֲ������
    printf("\n=== ���ȷֲ������ (0��%.1f) ===\n", uniform_upper);
    double uniform[10000];
    
    for (int i = 0; i < count; i++) {
        uniform[i] = uniform_random(uniform_upper);
    }
    
    // ����ͳ����
    double uniform_avg = mean(uniform, count);
    double uniform_std = stddev(uniform, count, uniform_avg);
    
    printf("������%d��0��%.1f֮��ľ��ȷֲ������\n", count, uniform_upper);
    printf("ƽ��ֵ: %.4f (����ֵ: %.4f)\n", uniform_avg, uniform_upper/2.0);
    printf("��׼��: %.4f (����ֵ: %.4f)\n", uniform_std, uniform_upper/sqrt(12));
    
    // ���ȷֲ���������
    int is_uniform = chi_square_test(uniform, count, 0, uniform_upper, bins);
    printf("�����������: ");
    if (is_uniform) {
        printf("����ͨ�������Լ��� (��5%%������ˮƽ��)\n");
    } else {
        printf("����δͨ�������Լ��飬���ܲ��Ǿ��ȷֲ�\n");
    }
    
    // ��ӡ���ȷֲ�ֱ��ͼ
    print_enhanced_histogram(uniform, count, bins, "���ȷֲ�ֱ��ͼ");
    
    // ������̬�ֲ������
    printf("\n=== ��̬�ֲ������ (��ֵ%.1f����׼��%.1f) ===\n", normal_mean, normal_stddev);
    double normal[10000];
    
    for (int i = 0; i < count; i++) {
        normal[i] = normal_random(normal_mean, normal_stddev);
    }
    
    // ����ͳ����
    double sample_mean = mean(normal, count);
    double sample_std = stddev(normal, count, sample_mean);
    double sample_median = median(normal, count);
    
    printf("������%d����̬�ֲ������\n", count);
    printf("������ֵ: %.4f (����ֵ: %.1f)\n", sample_mean, normal_mean);
    printf("������׼��: %.4f (����ֵ: %.1f)\n", sample_std, normal_stddev);
    printf("������λ��: %.4f (����ֵ: %.1f)\n", sample_median, normal_mean);
    
    // Lilliefors����
    int is_normal = lilliefors_test(normal, count, &p_value);
    printf("Lilliefors�������: ");
    if (is_normal) {
        printf("����ͨ����̬�Լ��� (��5%%������ˮƽ��)\n");
    } else {
        printf("����δͨ����̬�Լ��飬���ܲ�����̬�ֲ�\n");
    }
    
    // ��ӡ��̬�ֲ�ֱ��ͼ
    print_enhanced_histogram(normal, count, bins, "��̬�ֲ�ֱ��ͼ");
    
    
}	

int main() { 
while(1){
	printf("��ѡ����Ҫ���Ե�ͳ��ģ�飺\n"); 
	printf("1-������ɺ�ͳ��ÿ�����ĳ��ָ���\n");
	printf("2-������ɺ�ͳ��ÿ�����ظ����ֵļ���ķֲ����\n");
	printf("3-��������1������N�Ͳ�������������ֵ����\n") ;
	printf("4-��������2������N�Ͳ������������γ̲���\n");
	printf("5-��������3������N�Ͳ�����������������Բ���\n");
	printf("6-���Բ���4������N�Ͳ�����������Ƶ�ʲ���\n");
	printf("7-���Բ���5������N�Ͳ������������ۼӲ���\n");
	printf("8-��������ƹ�ʽ��������þ��ȷֲ������������̬�ֲ���������������ԡ�ģ��\n"); 
	printf("9-�˳�\n");
	int choose=0;
	scanf("%d",&choose);
	switch (choose) {
    case 1:
        count_frequency();
        printf("\n");
        break;
    case 2:
	    count_repetition_intervals();
	    printf("\n");
        break;
    case 3:
        entropy_test();
        printf("\n");
        break;
    case 4:
	    runs_test();
	    printf("\n");
        break;
    case 5:
        autocorrelation_test();
        printf("\n");
        break;
    case 6:
    	frequency_test();
    	printf("\n");
        break;
    case 7:
        cumulative_sums_test(); 
        printf("\n");
        break;
    case 8:
	    last_part_test();
	    printf("\n");
        break;
    case 9:
	    return 0; 
    default:
        printf("����������\n");
		printf("\n"); 
        break;
}
	
}  
    return 0;
}

