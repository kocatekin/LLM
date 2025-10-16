async function generate() {
      document.getElementById('resultSpan').innerText = "Loading...";
      //const topic = document.getElementById('topic').value;
      //const tone = document.getElementById('tone').value;
      //console.log(topic, tone);
      const myName = "wodprompt";
      const response = await fetch('http://localhost:5000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ myName })
      });
      //console.log(response);

      const data = await response.json();
      
      //document.getElementById('result').innerText = data || 'No answer returned.';
      document.getElementById("resultSpan").innerText = "";


      let wodTitle = document.getElementById("wodTitle");
      let rounds = document.getElementById("rounds");
      let mylist = document.getElementById("myList");

      wodTitle.innerText = data.title;
      rounds.innerText = `${data.rounds} rounds`;

      for(let i=0;i<data.workout.length;i++){
         mylist.innerHTML += `<li>${data.workout[i].reps} ${data.workout[i].movement}</li>`;
      }


      
      
    }