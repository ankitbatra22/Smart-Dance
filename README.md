# Smart-Dance (HTN 2021 Finalists)

### Link: 
https://devpost.com/software/smart-dance?ref_content=user-portfolio&ref_feature=in_progress

## Inspiration

With the rise of Tiktok and other social media platforms, pop dance culture has grown tremendously. As programmers, we wanted to find a way to give people a method of engaging with this new culture in a new and fun way.

## What it does 👀
Smart Dance uses pose estimation to store the key points of dances from popular dance videos and use those poses to guide the user on how to move their body to do the dance. Through our intuitive UI, users can find trending dances and learn or assess how well they're dancing when compared to their friends or internet stars that also do these dances. The score function we've made allows users to determine whether their pose is similar or different to the videos they are dancing along with. This makes dancing (and learning to dance) much more engaging as well as opens the door to the gamification of dancing with real-time feedback and encouragement (much like how the classic dance arcade games used to work). The only difference is not just enjoyable but is portable, easy to play and can be integrated with a lot more going forward!

## How we built it 💡
Smart-Dance uses Python on the backend, Mediapipe for real time pose estimation on CPUs and GPUs paired with OpenCV, as well as Flask, HTML / CSS, and MongoDB for the frontend-backend-integration.