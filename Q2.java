public class Q2 {
    public static void main(String[] args) {
        int [] array = {5,1,2,3,15,2,10,20};
        System.out.println(mma5MethodYourFirst(array));
    }

    //write a method called mma5MethodYourFirst that takes an array of integers as a parameter and returns the max, min, and average of the numbers divisible by 5 in the array.
    public static int mma5MethodYourFirst(int[] array) {
        int max = 0;
        int min = 0;
        int sum = 0;
        int average = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] % 5 == 0) {
                sum += array[i];
            }
            if (max < array[i]) {
                max = array[i];
            }
            if (min > array[i]) {
                min = array[i];
            }
                
            }
        }
        average = sum / 5;
        System.out.println("The max is " + max);
        System.out.println("The min is " + min);
        System.out.println("The average is " + average);
        return average;
    }
    
    public static int mma5MethodYourFirst(int [] x) {
        int max = x[0];
        int min = x[0];
        int avg = 0;

        for (int i = 0; i < x.length; i++) {
            if (max < x[i]) {
                max = x[i];
            }
            if (min > x[i]) {
                min = x[i];
            }
            avg += x[i];
        }
        avg = avg / x.length;
        System.out.println("Max: " + max);
        System.out.println("Min: " + min);
        System.out.println("Avg: " + avg);

        return max;
    }
}

// Output:
// Max: 20
// Min: 1
// Avg: 10
